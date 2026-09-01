import io
import os
import sys
import uuid
import base64
import threading
from datetime import datetime, timezone

# Vercel's Python runtime imports this file as a module rather than running it
# as a script, so the automatic sys.path[0] = script-dir behavior that makes
# `python api/index.py` resolve sibling imports (lung_segmentation, db, auth,
# etc.) does not happen there. Add this file's own directory explicitly so
# those sibling imports work in both environments.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from flask import (Flask, render_template, jsonify, request, abort, session,
                    redirect, url_for)
from dotenv import load_dotenv
from PIL import Image

# Load environment variables BEFORE importing the local modules below.
# db.py and study_store.py resolve DATABASE_PATH / STUDY_STORE_PATH at import
# time, so loading .env after them would silently leave those settings
# ineffective and fall back to the in-repo defaults.
load_dotenv()

from lung_segmentation import segment_lungs, SEGMENTATION_METHOD, SEGMENTATION_METHOD_VERSION  # noqa: E402
from mesh_reconstruction import (  # noqa: E402
    build_volume_geometry, build_lung_mesh, encode_typed_array, world_to_voxel_clamped,
    MeshReconstructionError, QUALITY_INTERACTIVE, QUALITY_HIGH_FIDELITY,
)
import db as dbmod          # noqa: E402
import auth as authmod      # noqa: E402
import cases as casemod     # noqa: E402
from study_store import StudyStore          # noqa: E402
from quantitative_analysis import analyze_study  # noqa: E402
from overlay_render import render_overlay_png, band_profile, OVERLAY_BANDS  # noqa: E402
import db_engine as dbengine     # noqa: E402
import measurements as measmod   # noqa: E402

# Initialize Flask app
# The template folder is pointed to the root /templates directory
app = Flask(__name__, template_folder='../templates')

# Session hardening + secret key (from SECRET_KEY env var). Returns False when
# no key is configured, in which case an ephemeral random key is used and
# sessions simply do not survive a restart - never a hardcoded fallback.
SECRET_KEY_CONFIGURED = authmod.configure_session_cookie(app)

# Create the schema on import so a fresh deployment/test run has its tables.
dbmod.init_db()
# Imaging-relational tables (studies/series/jobs/ROIs/measurements/annotations)
# - see api/models.py and ARCHITECTURE_AUDIT.md section 7 for why this is a
# separate module from db.py rather than a rewrite of it.
dbengine.init_models_db()


@app.teardown_appcontext
def _clear_request_doctor(exc=None):
    # g is per-request; nothing to close here (connections are per-thread and
    # reused), but keep the hook so the lifecycle is explicit.
    return None


# ---------------------------------------------------------------------------
# PUBLIC PAGES (no authentication)
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    """Premium public product website. Contains no patient data and no
    clinical functionality - marketing/educational content only."""
    return render_template('index.html')


@app.route('/technology')
def technology_page():
    """Public: the reconstruction pipeline, validation, and segmentation method."""
    return render_template('technology.html')


@app.route('/specifications')
def specifications_page():
    """Public: parameters, coordinate spaces, quality modes, measured performance."""
    return render_template('specifications.html')


@app.route('/safety')
def safety_page():
    """Public: boundaries, data handling, known limitations, FAQ."""
    return render_template('safety.html')


@app.route('/login', methods=['GET'])
def login_page():
    """Clean doctor sign-in page. Deliberately carries none of the public
    site's cinematic media."""
    if authmod.current_doctor() is not None:
        return redirect(url_for('cases_page'))
    return render_template('login.html', next=request.args.get('next', ''))


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Validates credentials server-side and establishes a session."""
    payload = request.get_json(silent=True) or request.form
    email = (payload.get('email') or '').strip()
    password = payload.get('password') or ''

    doctor = authmod.verify_credentials(email, password)
    if doctor is None:
        dbmod.record_audit('login_failed', doctor_id=None, target_type='email',
                            target_id=email[:120], outcome='denied', ip=authmod.client_ip())
        # Identical message for unknown account and wrong password.
        return jsonify({'status': 'FAIL', 'message': 'Invalid email or password.'}), 401

    authmod.login_session(doctor)
    dbmod.record_audit('login', doctor_id=doctor['id'], ip=authmod.client_ip())
    return jsonify({'status': 'OK', 'display_name': doctor['display_name'],
                    'redirect': '/cases'}), 200


@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    doctor = authmod.current_doctor()
    if doctor is not None:
        dbmod.record_audit('logout', doctor_id=doctor['id'], ip=authmod.client_ip())
    authmod.logout_session()
    return jsonify({'status': 'OK'}), 200


@app.route('/logout')
def logout_redirect():
    doctor = authmod.current_doctor()
    if doctor is not None:
        dbmod.record_audit('logout', doctor_id=doctor['id'], ip=authmod.client_ip())
    authmod.logout_session()
    return redirect(url_for('home'))


# ---------------------------------------------------------------------------
# CLINICAL PAGES (authentication required)
# ---------------------------------------------------------------------------

@app.route('/cases')
@authmod.login_required
def cases_page():
    """Doctor case list - the landing page after sign-in."""
    doctor = authmod.current_doctor()
    return render_template('cases.html', doctor_name=doctor['display_name'])


@app.route('/cases/<case_ref>')
@authmod.login_required
def case_workspace_page(case_ref):
    """Case workspace. Authorization is enforced here, not in the template."""
    doctor = authmod.current_doctor()
    case_row, reason = authmod.authorize_case_or_none(case_ref, doctor['id'])
    if case_row is None:
        dbmod.record_audit('case_access_denied', doctor_id=doctor['id'],
                            target_type='case_ref', target_id=case_ref,
                            outcome='denied', ip=authmod.client_ip())
        # Same 404 for "does not exist" and "exists but not yours", so a case
        # ref cannot be probed for existence.
        abort(404)
    dbmod.record_audit('case_accessed', doctor_id=doctor['id'],
                        target_type='case', target_id=case_row['id'], ip=authmod.client_ip())
    return render_template('case_workspace.html',
                            doctor_name=doctor['display_name'],
                            case=casemod.case_to_dict(case_row))


@app.route('/dashboard')
@authmod.login_required
def dashboard():
    """DICOM CT import workflow. Now behind authentication."""
    doctor = authmod.current_doctor()
    return render_template('dashboard.html', doctor_name=doctor['display_name'])


@app.route('/viewer/<study_id>')
@authmod.login_required
def viewer(study_id):
    """Renders the multi-planar CT viewer for a previously imported study.

    Ownership is checked here: a signed-in doctor may only open a study they
    imported. An unknown study id still renders the page (the frontend shows
    a clear "study not found" state), but a study belonging to someone else
    is refused outright.
    """
    doctor = authmod.current_doctor()
    study = STUDIES.get(study_id)
    if study is not None and study.get('owner_doctor_id') != doctor['id']:
        dbmod.record_audit('study_access_denied', doctor_id=doctor['id'],
                            target_type='study', target_id=study_id,
                            outcome='denied', ip=authmod.client_ip())
        abort(404)
    if study is not None:
        dbmod.record_audit('imaging_study_opened', doctor_id=doctor['id'],
                            target_type='study', target_id=study_id, ip=authmod.client_ip())
    return render_template('viewer.html', study_id=study_id,
                            doctor_name=doctor['display_name'])


# ---------------------------------------------------------------------------
# DICOM CT IMAGING FOUNDATION
# ---------------------------------------------------------------------------
# Feature 1: multi-file DICOM series upload -> validation -> spatial slice
#            ordering -> volume reconstruction -> Hounsfield unit conversion
#            -> technical study summary.
# Feature 2: axial / coronal / sagittal viewer endpoints (windowed slice
#            rendering + per-voxel HU inspection) backing templates/viewer.html.
#
# Privacy: uploaded DICOM files are parsed in-memory only and are never
# written to disk or sent to any external (including AI/LLM) service. Only
# an explicit allow-list of non-identifying technical fields is ever read
# out of a dataset for use in responses (see ALLOWED_SUMMARY_TAGS below) -
# patient-identifying fields (PatientName, PatientID, PatientBirthDate, etc.)
# are never accessed. No HIPAA/FDA/clinical-validation claims are made
# anywhere in this app.
#
# Storage note: reconstructed volumes are kept in an in-memory dict
# (STUDIES). This is intentional for this stage and is simple + fast for a
# single long-lived process (e.g. `python api/index.py` locally). On
# Vercel's serverless Python runtime, function instances are not guaranteed
# to persist between requests, so a study created on one invocation may not
# be visible on another - see the limitations note in the project README /
# final task summary.
# ---------------------------------------------------------------------------

# Disk-backed, memory-bounded study storage (see api/study_store.py).
# Studies now survive a process restart, and only the few most recently used
# volumes stay resident in RAM - the rest are memory-mapped on demand.
STUDIES = StudyStore()
STUDIES_LOCK = threading.Lock()


def _get_owned_study_or_error(study_id):
    """Resolves a study id to a study the signed-in doctor owns.

    Returns (study, None) on success or (None, flask_response) on failure.
    A study owned by another doctor returns the SAME 404 as a missing study,
    so study ids cannot be probed for existence by editing the URL.
    """
    doctor = authmod.current_doctor()
    study = STUDIES.get(study_id)
    if study is None:
        return None, (jsonify({'status': 'NOT_FOUND',
                                'message': 'Study not found. It may have expired or the server restarted.'}), 404)
    if study.get('owner_doctor_id') != doctor['id']:
        dbmod.record_audit('study_access_denied', doctor_id=doctor['id'],
                            target_type='study', target_id=study_id,
                            outcome='denied', ip=authmod.client_ip())
        return None, (jsonify({'status': 'NOT_FOUND',
                                'message': 'Study not found. It may have expired or the server restarted.'}), 404)
    return study, None

# Non-identifying technical fields only. Never add patient-identifying tags
# (PatientName, PatientID, PatientBirthDate, PatientSex, PatientAge,
# OtherPatientIDs, InstitutionAddress, ReferringPhysicianName, ...) to this
# list.
ALLOWED_SUMMARY_TAGS = [
    'Modality',
    'Manufacturer',
    'ManufacturerModelName',
    'KVP',
    'SliceThickness',
    'PhotometricInterpretation',
]

WINDOW_PRESETS = {
    'lung': {'ww': 1500, 'wl': -600},
    'soft_tissue': {'ww': 400, 'wl': 40},
    'bone': {'ww': 1800, 'wl': 400},
}

REQUIRED_UPLOAD_FIELD_NAME = 'files'

# ---------------------------------------------------------------------------
# ZIP archive support
# ---------------------------------------------------------------------------
# Public research collections (TCIA, and most others) distribute a series as a
# single .zip, and browsers cannot upload a folder on every platform. Accepting
# the archive directly removes the need to unpack hundreds of files by hand.
#
# Archives are expanded IN MEMORY only - nothing is written to disk - so a
# malicious relative path inside the archive has nothing to escape into. The
# limits below bound decompression so an archive cannot exhaust memory.
MAX_ARCHIVE_MEMBERS = 3000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024 ** 3       # 2 GB
ARCHIVE_EXTENSIONS = ('.zip',)
# Members that are never imaging data; skipping them keeps the per-file report
# readable instead of listing one FAIL per licence file.
SKIP_MEMBER_SUFFIXES = ('.txt', '.csv', '.xml', '.json', '.pdf', '.md', '.html',
                        '.xlsx', '.doc', '.docx', '.png', '.jpg', '.jpeg')


class _MemoryUpload:
    """Minimal stand-in for a Werkzeug FileStorage: filename + read()."""

    def __init__(self, filename, data):
        self.filename = filename
        self._data = data

    def read(self):
        return self._data


def expand_uploaded_archives(file_storages):
    """Replaces any .zip in the upload with its DICOM-looking members.

    Returns (expanded_files, notes). Non-archive uploads pass through
    untouched, so the existing loose-file workflow is unaffected.
    """
    import zipfile

    expanded = []
    notes = []

    for fs in file_storages:
        name = (fs.filename or '').lower()
        if not name.endswith(ARCHIVE_EXTENSIONS):
            expanded.append(fs)
            continue

        raw = fs.read()
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile:
            notes.append(f"'{fs.filename}' is not a readable ZIP archive and was skipped.")
            continue

        members = [i for i in zf.infolist() if not i.is_dir()]
        skipped_types = 0
        total_bytes = 0
        added = 0

        for info in members:
            member_name = os.path.basename(info.filename) or info.filename
            if member_name.startswith('.') or member_name.lower().endswith(SKIP_MEMBER_SUFFIXES):
                skipped_types += 1
                continue
            if added >= MAX_ARCHIVE_MEMBERS:
                notes.append(
                    f"'{fs.filename}' contains more than {MAX_ARCHIVE_MEMBERS} files; the rest were "
                    f"not read. Split the archive if the series is genuinely larger.")
                break
            if total_bytes + info.file_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                notes.append(
                    f"'{fs.filename}' exceeds the {MAX_ARCHIVE_UNCOMPRESSED_BYTES // 1024**3} GB "
                    f"uncompressed limit; reading stopped early.")
                break
            try:
                data = zf.read(info)
            except Exception:  # noqa: BLE001
                continue
            total_bytes += len(data)
            expanded.append(_MemoryUpload(member_name, data))
            added += 1

        notes.append(
            f"Extracted {added} file(s) from '{fs.filename}'" +
            (f"; skipped {skipped_types} non-imaging file(s)." if skipped_types else "."))

    return expanded, notes


# ---------------------------------------------------------------------------
# Helper: safe attribute read
# ---------------------------------------------------------------------------

def _get(ds, tag, default=None):
    """Read a DICOM attribute defensively, never raising on absence."""
    try:
        value = getattr(ds, tag, default)
    except Exception:
        return default
    return value


def _as_float_list(value):
    try:
        return [float(v) for v in value]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# validate_dicom_series
# ---------------------------------------------------------------------------

def validate_dicom_series(file_storages):
    """
    Reads and validates a list of uploaded files as a DICOM CT series.

    Returns:
        per_file_results: list of {filename, status, messages}
        series_result: {status, messages, series_instance_uid, study_instance_uid}
        good_datasets: list of entry dicts usable for reconstruction
                       (status PASS or WARNING, not FAIL)
    """
    per_file_results = []
    parsed = []  # list of dicts: {filename, ds, status, messages, ...}

    for fs in file_storages:
        filename = fs.filename or 'unnamed_file'
        messages = []
        status = 'PASS'

        raw = fs.read()
        if not raw:
            per_file_results.append({'filename': filename, 'status': 'FAIL',
                                      'messages': ['File is empty.']})
            continue

        try:
            ds = pydicom.dcmread(io.BytesIO(raw), force=True)
        except (InvalidDicomError, Exception) as exc:  # noqa: BLE001
            per_file_results.append({'filename': filename, 'status': 'FAIL',
                                      'messages': [f'Unable to read file as DICOM: {exc}']})
            continue

        # Modality
        modality = _get(ds, 'Modality')
        if modality is None:
            status = 'WARNING'
            messages.append('Modality tag is missing.')
        elif str(modality).upper() != 'CT':
            status = 'WARNING'
            messages.append(f"Modality is '{modality}', not CT.")

        # UIDs needed for series grouping
        study_uid = _get(ds, 'StudyInstanceUID')
        series_uid = _get(ds, 'SeriesInstanceUID')
        sop_uid = _get(ds, 'SOPInstanceUID')
        if not series_uid:
            status = 'WARNING'
            messages.append('SeriesInstanceUID is missing; file will be assumed to belong to this series.')
        if not study_uid:
            status = 'WARNING'
            messages.append('StudyInstanceUID is missing.')
        if not sop_uid:
            status = 'WARNING'
            messages.append('SOPInstanceUID is missing; duplicate detection is unreliable for this file.')

        # Rows / Columns
        rows = _get(ds, 'Rows')
        cols = _get(ds, 'Columns')
        if not rows or not cols:
            status = 'FAIL'
            messages.append('Rows/Columns are missing; cannot use this slice.')

        # Pixel data
        has_pixels = hasattr(ds, 'PixelData')
        pixel_array = None
        if has_pixels and status != 'FAIL':
            try:
                pixel_array = ds.pixel_array
            except Exception as exc:  # noqa: BLE001
                status = 'FAIL'
                messages.append(f'Pixel data could not be decoded: {exc}')
        elif not has_pixels:
            status = 'FAIL'
            messages.append('No pixel data present.')

        # PixelSpacing
        pixel_spacing = _as_float_list(_get(ds, 'PixelSpacing')) if _get(ds, 'PixelSpacing') else None
        if not pixel_spacing:
            if status == 'PASS':
                status = 'WARNING'
            messages.append('PixelSpacing is missing; a 1mm x 1mm fallback will be used for display scaling only.')

        # ImageOrientationPatient / ImagePositionPatient
        iop = _as_float_list(_get(ds, 'ImageOrientationPatient')) if _get(ds, 'ImageOrientationPatient') else None
        ipp = _as_float_list(_get(ds, 'ImagePositionPatient')) if _get(ds, 'ImagePositionPatient') else None
        if not iop or not ipp:
            if status == 'PASS':
                status = 'WARNING'
            messages.append('ImageOrientationPatient/ImagePositionPatient missing; spatial ordering fallback may be needed.')

        # SliceThickness
        if _get(ds, 'SliceThickness') is None:
            messages.append('SliceThickness: NOT AVAILABLE.')

        # RescaleSlope / RescaleIntercept
        if _get(ds, 'RescaleSlope') is None or _get(ds, 'RescaleIntercept') is None:
            messages.append('RescaleSlope/RescaleIntercept missing; HU conversion NOT AVAILABLE for this slice.')

        # PhotometricInterpretation
        if _get(ds, 'PhotometricInterpretation') is None:
            if status == 'PASS':
                status = 'WARNING'
            messages.append('PhotometricInterpretation is missing.')

        if not messages:
            messages.append('All checked fields present and consistent.')

        per_file_results.append({'filename': filename, 'status': status, 'messages': messages})
        parsed.append({
            'filename': filename, 'ds': ds, 'status': status, 'messages': messages,
            'study_uid': study_uid, 'series_uid': series_uid, 'sop_uid': sop_uid,
            'rows': rows, 'cols': cols, 'pixel_array': pixel_array,
        })

    series_messages = []
    series_status = 'PASS'

    usable = [p for p in parsed if p['status'] != 'FAIL']

    if not usable:
        series_status = 'FAIL'
        series_messages.append('No usable DICOM slices were found in the upload.')
        return per_file_results, {'status': series_status, 'messages': series_messages,
                                   'series_instance_uid': None, 'study_instance_uid': None}, []

    # Determine dominant series/study UID (most common non-empty value)
    def _dominant(key):
        values = [p[key] for p in usable if p[key]]
        if not values:
            return None
        return max(set(values), key=values.count)

    dominant_series_uid = _dominant('series_uid')
    dominant_study_uid = _dominant('study_uid')

    mismatched = [p for p in usable if p['series_uid'] and dominant_series_uid and p['series_uid'] != dominant_series_uid]
    if mismatched:
        series_status = 'WARNING'
        series_messages.append(
            f"{len(mismatched)} file(s) have a different SeriesInstanceUID than the majority "
            f"and were excluded from the reconstructed volume."
        )
        usable = [p for p in usable if p not in mismatched]

    # Duplicate SOPInstanceUID detection
    seen_sop = {}
    deduped = []
    dup_count = 0
    for p in usable:
        key = p['sop_uid'] or p['filename']
        if key in seen_sop:
            dup_count += 1
            continue
        seen_sop[key] = True
        deduped.append(p)
    usable = deduped
    if dup_count:
        series_status = 'WARNING'
        series_messages.append(f'{dup_count} duplicate SOPInstanceUID file(s) were detected and dropped.')

    # Inconsistent dimensions
    dims = {(p['rows'], p['cols']) for p in usable}
    if len(dims) > 1:
        dominant_dim = max(dims, key=lambda d: sum(1 for p in usable if (p['rows'], p['cols']) == d))
        off_dim = [p for p in usable if (p['rows'], p['cols']) != dominant_dim]
        usable = [p for p in usable if (p['rows'], p['cols']) == dominant_dim]
        series_status = 'WARNING'
        series_messages.append(
            f'{len(off_dim)} slice(s) had inconsistent Rows/Columns and were excluded from the volume.'
        )

    if len(usable) < 2:
        series_status = 'FAIL'
        series_messages.append('Fewer than 2 consistent slices remain; a volume cannot be reconstructed.')

    if not series_messages:
        series_messages.append('Series-level checks passed.')

    series_result = {
        'status': series_status,
        'messages': series_messages,
        'series_instance_uid': dominant_series_uid,
        'study_instance_uid': dominant_study_uid,
    }

    good_datasets = usable if series_status != 'FAIL' else []
    return per_file_results, series_result, good_datasets


# ---------------------------------------------------------------------------
# order_slices_spatially
# ---------------------------------------------------------------------------

def order_slices_spatially(entries):
    """
    Orders parsed DICOM entries (dicts with a 'ds' key) by physical slice
    position, computed from ImageOrientationPatient / ImagePositionPatient.
    Falls back to SliceLocation, then InstanceNumber, if orientation/position
    data isn't usable - the fallback is always flagged back to the caller.

    Returns: ordered_entries, fallback_used (bool), fallback_method (str|None),
             slice_positions (list[float] aligned with ordered_entries),
             warnings (list[str])
    """
    warnings = []
    can_use_geometry = True
    projections = []

    for e in entries:
        iop = _as_float_list(_get(e['ds'], 'ImageOrientationPatient'))
        ipp = _as_float_list(_get(e['ds'], 'ImagePositionPatient'))
        if not iop or len(iop) != 6 or not ipp or len(ipp) != 3:
            can_use_geometry = False
            break
        row_cosines = np.array(iop[0:3], dtype=np.float64)
        col_cosines = np.array(iop[3:6], dtype=np.float64)
        normal = np.cross(row_cosines, col_cosines)
        projections.append(float(np.dot(np.array(ipp, dtype=np.float64), normal)))

    if can_use_geometry:
        order = sorted(range(len(entries)), key=lambda i: projections[i])
        ordered_entries = [entries[i] for i in order]
        slice_positions = [projections[i] for i in order]
        return ordered_entries, False, None, slice_positions, warnings

    # Fallback 1: SliceLocation
    slice_locations = [_get(e['ds'], 'SliceLocation') for e in entries]
    if all(sl is not None for sl in slice_locations):
        order = sorted(range(len(entries)), key=lambda i: float(slice_locations[i]))
        ordered_entries = [entries[i] for i in order]
        slice_positions = [float(slice_locations[i]) for i in order]
        warnings.append('Spatial ordering fallback used: sorted by SliceLocation because '
                         'ImageOrientationPatient/ImagePositionPatient were unavailable.')
        return ordered_entries, True, 'SliceLocation', slice_positions, warnings

    # Fallback 2: InstanceNumber
    instance_numbers = [_get(e['ds'], 'InstanceNumber') for e in entries]
    if all(n is not None for n in instance_numbers):
        order = sorted(range(len(entries)), key=lambda i: int(instance_numbers[i]))
        ordered_entries = [entries[i] for i in order]
        slice_positions = [float(instance_numbers[i]) for i in order]
        warnings.append('Spatial ordering fallback used: sorted by InstanceNumber because neither '
                         'geometry nor SliceLocation were available. Spacing may not reflect true '
                         'physical distance.')
        return ordered_entries, True, 'InstanceNumber', slice_positions, warnings

    # Last resort: keep upload order
    warnings.append('No reliable ordering metadata found (geometry, SliceLocation, or InstanceNumber). '
                     'Slices were kept in upload order; ordering may not reflect true anatomy.')
    slice_positions = list(range(len(entries)))
    return entries, True, 'upload_order', slice_positions, warnings


# ---------------------------------------------------------------------------
# build_volume / convert_to_hu
# ---------------------------------------------------------------------------

def build_volume(ordered_entries):
    """Stacks per-slice pixel arrays into a [slice, row, column] volume."""
    arrays = [e['pixel_array'] for e in ordered_entries]
    volume = np.stack(arrays).astype(np.float32)
    return volume


def convert_to_hu(volume, ordered_entries, in_place=False):
    """
    Applies HU = pixel_value * RescaleSlope + RescaleIntercept per-slice.
    Slope/intercept are read per dataset, never assumed constant across the
    series. Slices missing either value are left as raw pixel values and
    flagged as HU-unavailable for that slice.

    `in_place=True` rewrites `volume` rather than allocating a second
    full-size array. On a real chest CT that array is hundreds of megabytes,
    and the caller in upload_dicom_series discards the raw volume immediately
    afterwards, so the copy is pure overhead there. The default stays False
    so existing callers keep the non-destructive behaviour.
    """
    hu_volume = volume if in_place else np.empty_like(volume, dtype=np.float32)
    hu_available_per_slice = []

    for i, e in enumerate(ordered_entries):
        slope = _get(e['ds'], 'RescaleSlope')
        intercept = _get(e['ds'], 'RescaleIntercept')
        if slope is not None and intercept is not None:
            try:
                # multiply/add with out= avoids a per-slice temporary
                np.multiply(volume[i], float(slope), out=hu_volume[i])
                np.add(hu_volume[i], float(intercept), out=hu_volume[i])
                hu_available_per_slice.append(True)
                continue
            except Exception:  # noqa: BLE001
                pass
        if not in_place:
            hu_volume[i] = volume[i]
        hu_available_per_slice.append(False)

    return hu_volume, hu_available_per_slice


def _release_slice_pixel_data(ordered_entries):
    """Frees decoded pixel data once it has been copied into the volume.

    validate_dicom_series keeps every decoded `pixel_array` AND every parsed
    dataset (each holding its raw PixelData bytes) alive so the volume can be
    stacked. After build_volume that is a duplicate of data now living in the
    volume - on a 120-slice 512x512 study, ~125 MB of it. The datasets stay
    alive because later stages still read non-pixel tags (RescaleSlope,
    PixelSpacing, ImageOrientationPatient, ...); only the pixel payload goes.
    """
    for e in ordered_entries:
        e['pixel_array'] = None
        ds = e.get('ds')
        if ds is None:
            continue
        try:
            if 'PixelData' in ds:
                del ds.PixelData
        except Exception:  # noqa: BLE001
            pass
        # pydicom caches the decoded array on the dataset; drop it too.
        for attr in ('_pixel_array', '_pixel_id'):
            if hasattr(ds, attr):
                try:
                    setattr(ds, attr, None)
                except Exception:  # noqa: BLE001
                    pass


# ---------------------------------------------------------------------------
# build_study_summary
# ---------------------------------------------------------------------------

def build_study_summary(ordered_entries, slice_positions, fallback_used, fallback_method,
                         hu_available_per_slice, series_result, validation_warnings):
    first_ds = ordered_entries[0]['ds']
    rows = int(_get(first_ds, 'Rows'))
    cols = int(_get(first_ds, 'Columns'))
    n_slices = len(ordered_entries)

    pixel_spacing = _as_float_list(_get(first_ds, 'PixelSpacing')) or [1.0, 1.0]

    diffs = np.diff(np.array(slice_positions, dtype=np.float64))
    abs_diffs = np.abs(diffs)
    if len(abs_diffs) > 0:
        slice_spacing = float(np.median(abs_diffs))
        spacing_irregular = bool(np.any(np.abs(abs_diffs - slice_spacing) > 0.2 * max(slice_spacing, 1e-6)))
        large_gap = bool(np.any(abs_diffs > 3 * max(slice_spacing, 1e-6)))
    else:
        slice_spacing = None
        spacing_irregular = False
        large_gap = False

    slice_thickness = _get(first_ds, 'SliceThickness')

    orientation = _as_float_list(_get(first_ds, 'ImageOrientationPatient'))
    orientation_status = 'AVAILABLE' if (orientation and not fallback_used) else (
        f'FALLBACK ({fallback_method})' if fallback_used else 'NOT AVAILABLE'
    )

    hu_all = all(hu_available_per_slice)
    hu_none = not any(hu_available_per_slice)
    hu_status = 'AVAILABLE' if hu_all else ('PARTIAL' if not hu_none else 'NOT AVAILABLE')

    scanner_metadata = {}
    for tag in ALLOWED_SUMMARY_TAGS:
        val = _get(first_ds, tag)
        if val is not None:
            scanner_metadata[tag] = str(val)

    warnings = list(validation_warnings)
    if spacing_irregular:
        warnings.append('Irregular spacing detected between slices.')
    if large_gap:
        warnings.append('A large gap was detected between two or more consecutive slices.')

    return {
        'slice_count': n_slices,
        'rows': rows,
        'columns': cols,
        'volume_dimensions': {'slices': n_slices, 'rows': rows, 'columns': cols},
        'pixel_spacing_mm': pixel_spacing,
        'slice_spacing_mm': slice_spacing,
        'slice_thickness_mm': float(slice_thickness) if slice_thickness is not None else None,
        'orientation_status': orientation_status,
        'hu_conversion_status': hu_status,
        'series_status': series_result['status'],
        'validation_warnings': warnings,
        'scanner_metadata': scanner_metadata,
        'window_presets': WINDOW_PRESETS,
    }


# ---------------------------------------------------------------------------
# Windowed slice rendering
# ---------------------------------------------------------------------------

def _extract_plane(hu_volume, plane, index):
    n_slices, rows, cols = hu_volume.shape
    if plane == 'axial':
        if not (0 <= index < n_slices):
            return None, None
        return hu_volume[index, :, :], n_slices
    if plane == 'coronal':
        if not (0 <= index < rows):
            return None, None
        return hu_volume[:, index, :], rows
    if plane == 'sagittal':
        if not (0 <= index < cols):
            return None, None
        return hu_volume[:, :, index], cols
    return None, None


def _window_to_png_base64(plane_array, ww, wl, aspect_ratio=1.0):
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(plane_array, lo, hi)
    normalized = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)

    img = Image.fromarray(normalized, mode='L')

    # Correct display aspect ratio for coronal/sagittal planes where the
    # through-plane (slice) spacing usually differs from in-plane pixel
    # spacing. aspect_ratio = physical_height / physical_width per source
    # pixel; only the vertical dimension is rescaled here for simplicity.
    if aspect_ratio and abs(aspect_ratio - 1.0) > 0.01:
        new_h = max(1, int(round(img.height * aspect_ratio)))
        img = img.resize((img.width, new_h), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


# ---------------------------------------------------------------------------
# API: upload
# ---------------------------------------------------------------------------

@app.route('/api/dicom/upload', methods=['POST'])
@authmod.api_login_required
def upload_dicom_series():
    """
    Accepts a multi-file DICOM series upload (multipart/form-data, field
    name 'files'), validates it, spatially orders the slices, reconstructs
    a volume, converts to Hounsfield units where possible, and returns a
    validation report + technical study summary + a study_id to open in the
    viewer.
    """
    files = request.files.getlist(REQUIRED_UPLOAD_FIELD_NAME)
    if not files:
        return jsonify({'status': 'FAIL', 'message': 'No files were uploaded. Expected multipart field "files".'}), 400

    # A .zip is expanded in memory so public datasets can be imported as the
    # single archive they are distributed as.
    files, archive_notes = expand_uploaded_archives(files)
    if not files:
        return jsonify({'status': 'FAIL',
                         'message': 'The upload contained no readable files.',
                         'archive_notes': archive_notes}), 400

    per_file_results, series_result, good_entries = validate_dicom_series(files)

    if series_result['status'] == 'FAIL' or not good_entries:
        return jsonify({
            'status': 'FAIL',
            'per_file_results': per_file_results,
            'series_result': series_result,
            'archive_notes': archive_notes,
        }), 422

    ordered_entries, fallback_used, fallback_method, slice_positions, order_warnings = \
        order_slices_spatially(good_entries)

    volume = build_volume(ordered_entries)
    # The per-slice pixel data is now duplicated inside `volume`; release it
    # before allocating anything else (see _release_slice_pixel_data).
    _release_slice_pixel_data(ordered_entries)
    # in_place: `volume` is discarded right after this call, so converting it
    # in place avoids a second full-size allocation.
    hu_volume, hu_available_per_slice = convert_to_hu(volume, ordered_entries, in_place=True)
    volume = None

    all_warnings = series_result['messages'] + order_warnings
    summary = build_study_summary(
        ordered_entries, slice_positions, fallback_used, fallback_method,
        hu_available_per_slice, series_result, all_warnings,
    )

    # Geometry for the 3D reconstruction pipeline: the DICOM patient-space
    # origin/orientation of the volume, only trusted when real (non-fallback)
    # slice-ordering geometry was available - see mesh_reconstruction.py.
    first_ds = ordered_entries[0]['ds']
    orientation_reliable = not fallback_used
    origin_mm = _as_float_list(_get(first_ds, 'ImagePositionPatient')) if orientation_reliable else None
    iop = _as_float_list(_get(first_ds, 'ImageOrientationPatient')) if orientation_reliable else None
    geometry = build_volume_geometry(
        shape=hu_volume.shape,
        origin_mm=origin_mm,
        pixel_spacing_row_col=summary['pixel_spacing_mm'],
        slice_spacing_mm=summary['slice_spacing_mm'],
        iop=iop,
        orientation_reliable=orientation_reliable,
    )

    doctor = authmod.current_doctor()
    study_id = str(uuid.uuid4())
    with STUDIES_LOCK:
        STUDIES[study_id] = {
            'hu_volume': hu_volume,
            'hu_available_per_slice': hu_available_per_slice,
            'pixel_spacing': summary['pixel_spacing_mm'],
            'slice_spacing': summary['slice_spacing_mm'],
            'summary': summary,
            'geometry': geometry,
            'segmentation': None,
            # Ownership: every subsequent read of this study re-checks this
            # against the signed-in doctor (see _get_owned_study_or_error).
            'owner_doctor_id': doctor['id'],
            'created_at': datetime.now(timezone.utc).isoformat(),
        }

    dbmod.record_audit('imaging_study_imported', doctor_id=doctor['id'],
                        target_type='study', target_id=study_id, ip=authmod.client_ip())

    # Optionally attach the freshly imported study to a case the doctor owns.
    case_ref = request.form.get('case_ref')
    linked_case_id = None
    if case_ref:
        case_row, _reason = authmod.authorize_case_or_none(case_ref, doctor['id'])
        if case_row is not None:
            casemod.attach_study(case_row['id'], study_id)
            casemod.update_case_status(case_row['id'], 'ready')
            linked_case_id = case_row['id']

    # Relational index of the import (ImagingStudy/ImagingSeries + an IMPORT
    # job record) - additive bookkeeping alongside the file-based study store,
    # which remains the source of truth for the actual pixel data. A failure
    # here must not break a successful import; it is recorded, not raised.
    try:
        job_id = measmod.start_job(study_id, "IMPORT", method="dicom_upload")
        measmod.upsert_imaging_study(
            study_id, owner_doctor_id=doctor['id'], case_id=linked_case_id,
            modality=summary.get('scanner_metadata', {}).get('Modality'),
            slice_count=hu_volume.shape[0], rows=hu_volume.shape[1], columns=hu_volume.shape[2],
        )
        measmod.add_series(
            study_id,
            slice_count=hu_volume.shape[0], rows=hu_volume.shape[1], columns=hu_volume.shape[2],
            pixel_spacing_row_mm=(summary.get('pixel_spacing_mm') or [None, None])[0],
            pixel_spacing_col_mm=(summary.get('pixel_spacing_mm') or [None, None])[1],
            slice_spacing_mm=summary.get('slice_spacing_mm'),
            orientation_reliable=geometry.orientation_reliable,
            hu_available=any(hu_available_per_slice),
            fallback_ordering_used=fallback_used,
        )
        measmod.complete_job(job_id)
    except Exception as exc:  # noqa: BLE001 - see comment above
        dbmod.record_audit('imaging_study_relational_index_failed', doctor_id=doctor['id'],
                            target_type='study', target_id=study_id, outcome='failed',
                            ip=authmod.client_ip())

    return jsonify({
        'status': 'PASS' if series_result['status'] == 'PASS' else 'WARNING',
        'study_id': study_id,
        'per_file_results': per_file_results,
        'series_result': series_result,
        'summary': summary,
        'archive_notes': archive_notes,
    }), 200


@app.route('/api/dicom/study/<study_id>', methods=['DELETE'])
@authmod.api_login_required
def clear_study(study_id):
    """Clears/resets an imported study from memory. Ownership is verified
    first so one doctor cannot delete another's study by id."""
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err
    with STUDIES_LOCK:
        STUDIES.pop(study_id, None)
    return jsonify({'status': 'OK', 'message': 'Study cleared.'}), 200


@app.route('/api/dicom/study/<study_id>/summary', methods=['GET'])
@authmod.api_login_required
def get_study_summary(study_id):
    """Returns the technical study summary for a previously imported study."""
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err
    return jsonify({'status': 'OK', 'study_id': study_id, 'summary': study['summary']})


@app.route('/api/dicom/study/<study_id>/slice/<plane>/<int:index>', methods=['GET'])
@authmod.api_login_required
def get_reconstructed_slice(study_id, plane, index):
    """
    Returns a single windowed slice (axial / coronal / sagittal) from the
    reconstructed volume as a base64 PNG, honoring window width / window
    level query params (ww, wl) without modifying the underlying HU volume.
    """
    if plane not in ('axial', 'coronal', 'sagittal'):
        return jsonify({'status': 'FAIL', 'message': 'plane must be axial, coronal, or sagittal.'}), 400

    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err

    hu_volume = study['hu_volume']
    plane_array, total = _extract_plane(hu_volume, plane, index)
    if plane_array is None:
        return jsonify({'status': 'FAIL', 'message': f'Index {index} out of range for {plane} (0..{total - 1 if total else 0}).'}), 400

    preset = request.args.get('preset')
    if preset and preset in WINDOW_PRESETS:
        ww = WINDOW_PRESETS[preset]['ww']
        wl = WINDOW_PRESETS[preset]['wl']
    else:
        try:
            ww = float(request.args.get('ww', WINDOW_PRESETS['lung']['ww']))
            wl = float(request.args.get('wl', WINDOW_PRESETS['lung']['wl']))
        except ValueError:
            return jsonify({'status': 'FAIL', 'message': 'ww and wl must be numeric.'}), 400

    pixel_spacing = study['pixel_spacing'] or [1.0, 1.0]
    slice_spacing = study['slice_spacing'] or pixel_spacing[0]

    aspect_ratio = 1.0
    if plane in ('coronal', 'sagittal') and pixel_spacing[0]:
        aspect_ratio = slice_spacing / pixel_spacing[0]

    # Optional density-classification overlay. Requires a segmentation: the
    # classification is only meaningful inside lung tissue, and tinting
    # non-lung structures would be actively misleading.
    overlay_requested = request.args.get('overlay') in ('1', 'true', 'yes')
    overlay_status = None
    if overlay_requested:
        seg = study.get('segmentation')
        if seg is None or not getattr(seg, 'success', False) or seg.mask is None:
            overlay_status = 'NOT_AVAILABLE: run lung segmentation first.'
        else:
            mask_plane, _t = _extract_plane(seg.mask, plane, index)
            if mask_plane is None:
                overlay_status = 'NOT_AVAILABLE: mask does not cover this plane index.'
            else:
                png_b64 = render_overlay_png(plane_array, mask_plane.astype(bool),
                                              ww, wl, aspect_ratio)
                overlay_status = 'OK'

    if overlay_status != 'OK':
        png_b64 = _window_to_png_base64(plane_array, ww, wl, aspect_ratio)

    n_slices, rows, cols = hu_volume.shape
    hu_available = study['hu_available_per_slice']
    if plane == 'axial':
        hu_available_here = hu_available[index]
    else:
        hu_available_here = any(hu_available)

    return jsonify({
        'status': 'OK',
        'plane': plane,
        'index': index,
        'total': total,
        'ww': ww,
        'wl': wl,
        'hu_available': hu_available_here,
        'image_base64': png_b64,
        'volume_shape': {'slices': n_slices, 'rows': rows, 'columns': cols},
        # Vertical scale factor applied to the raw plane before PNG encoding
        # (1.0 for axial). The frontend needs this to map a click on the
        # rendered image back to true volume voxel coordinates.
        'aspect_ratio': aspect_ratio,
        'overlay': overlay_status,
    })


@app.route('/api/dicom/study/<study_id>/voxel', methods=['GET'])
@authmod.api_login_required
def get_voxel_value(study_id):
    """
    Returns the HU value at a specific (x, y, z) voxel, where x is the
    column index, y is the row index, and z is the slice (axial) index -
    matching the axes used by the /slice endpoint. Never invents a value:
    if HU conversion wasn't available for that slice, hu is returned as null
    with hu_available=false.
    """
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err

    try:
        x = int(request.args['x'])
        y = int(request.args['y'])
        z = int(request.args['z'])
    except (KeyError, ValueError):
        return jsonify({'status': 'FAIL', 'message': 'x, y, and z query params are required and must be integers.'}), 400

    hu_volume = study['hu_volume']
    n_slices, rows, cols = hu_volume.shape
    if not (0 <= x < cols and 0 <= y < rows and 0 <= z < n_slices):
        return jsonify({'status': 'FAIL', 'message': 'Voxel coordinates out of range.'}), 400

    hu_available = study['hu_available_per_slice'][z]
    hu_value = float(hu_volume[z, y, x]) if hu_available else None

    return jsonify({
        'status': 'OK',
        'x': x, 'y': y, 'z': z,
        'hu_available': hu_available,
        'hu': hu_value,
    })


# ---------------------------------------------------------------------------
# 3D LUNG RECONSTRUCTION & VISUALIZATION
# ---------------------------------------------------------------------------
# Pipeline: segment_lungs() (api/lung_segmentation.py, rule-based HU/
# connected-component segmentation - NOT AI) -> build_lung_mesh()
# (api/mesh_reconstruction.py, marching cubes using true physical voxel
# spacing). Segmentation runs once per study and is cached in-memory;
# reconstruction can be re-run per quality/part without re-segmenting.
# Volume-texture serves a (possibly downsampled, see MAX volume-texture dim
# below) copy of the real HU volume for direct volume rendering - never a
# generic/prebuilt model.
# ---------------------------------------------------------------------------

VOLUME_TEXTURE_DEFAULT_MAX_DIM = 160
VOLUME_TEXTURE_HARD_CAP_DIM = 256


@app.route('/api/dicom/study/<study_id>/geometry', methods=['GET'])
@authmod.api_login_required
def get_study_geometry(study_id):
    """Returns the DICOM-derived volume<->patient-space affine transform
    (see mesh_reconstruction.VolumeGeometry) so the frontend can convert
    between mesh/world coordinates and volume voxel indices itself."""
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err
    return jsonify({'status': 'OK', 'geometry': study['geometry'].to_public_dict()})


@app.route('/api/dicom/study/<study_id>/segment-lungs', methods=['POST'])
@authmod.api_login_required
def segment_lungs_endpoint(study_id):
    """
    Runs the rule-based lung segmentation pipeline on the study's
    reconstructed HU volume and caches the result for subsequent
    /reconstruct3d calls. Returns quality metrics only - never the raw mask
    (not needed by the frontend, and unnecessarily large to transfer).
    """
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err

    geometry = study['geometry']
    result = segment_lungs(
        hu_volume=study['hu_volume'],
        hu_available_per_slice=study['hu_available_per_slice'],
        spacing_mm=geometry.spacing_mm,
        col_cosines=geometry.col_cosines,
        orientation_reliable=geometry.orientation_reliable,
    )

    with STUDIES_LOCK:
        study['segmentation'] = result
        # Persist so a later reconstruct3d call (possibly after a restart, or
        # after this study was evicted from the memory cache) does not have to
        # re-run segmentation.
        STUDIES.save_segmentation(study_id, result)

    try:
        job_id = measmod.start_job(study_id, "SEGMENTATION", method=result.method,
                                    method_version=result.method_version)
        if result.success:
            measmod.complete_job(job_id)
            measmod.update_study_status(study_id, "segmented")
        else:
            measmod.fail_job(job_id, "; ".join(result.warnings) or "Segmentation implausible.")
    except Exception:  # noqa: BLE001 - relational bookkeeping must not block the response
        pass

    http_status = 200 if result.success else 422
    return jsonify({**result.to_public_dict()}), http_status


@app.route('/api/dicom/study/<study_id>/reconstruct3d', methods=['GET'])
@authmod.api_login_required
def reconstruct3d_endpoint(study_id):
    """
    Builds a triangulated lung surface mesh from the cached segmentation
    result via marching cubes at true physical voxel spacing.

    Query params:
        quality: 'interactive' (downsampled, real-time) | 'high_fidelity'
                 (native resolution, scientific reference). Default: interactive.
        part: 'combined' (both lungs) | 'left' | 'right'. Default: combined.
    """
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err

    segmentation = study.get('segmentation')
    if segmentation is None:
        return jsonify({'status': 'FAIL', 'message': 'Run /segment-lungs for this study first.'}), 409
    if not segmentation.success:
        return jsonify({
            'status': 'FAIL',
            'message': 'Segmentation did not pass plausibility checks; 3D reconstruction refused.',
            'segmentation_warnings': segmentation.warnings,
        }), 422

    quality = request.args.get('quality', QUALITY_INTERACTIVE)
    if quality not in (QUALITY_INTERACTIVE, QUALITY_HIGH_FIDELITY):
        return jsonify({'status': 'FAIL', 'message': "quality must be 'interactive' or 'high_fidelity'."}), 400

    part = request.args.get('part', 'combined')
    if part == 'combined':
        mask = segmentation.mask
    elif part == 'left':
        mask = segmentation.left_mask
    elif part == 'right':
        mask = segmentation.right_mask
    else:
        return jsonify({'status': 'FAIL', 'message': "part must be 'combined', 'left', or 'right'."}), 400

    if mask is None:
        return jsonify({
            'status': 'FAIL',
            'message': f"'{part}' is not available for this study (left/right lungs could not be "
                       f"reliably distinguished - see segmentation warnings).",
            'segmentation_warnings': segmentation.warnings,
        }), 422

    job_id = None
    try:
        job_id = measmod.start_job(study_id, "MESH_GENERATION",
                                    parameters={"quality": quality, "part": part})
    except Exception:  # noqa: BLE001
        pass
    try:
        mesh = build_lung_mesh(mask, study['geometry'], quality=quality)
    except MeshReconstructionError as exc:
        if job_id:
            try:
                measmod.fail_job(job_id, str(exc))
            except Exception:  # noqa: BLE001
                pass
        return jsonify({'status': 'FAIL', 'message': f'3D reconstruction failed: {exc}'}), 422
    if job_id:
        try:
            measmod.complete_job(job_id)
        except Exception:  # noqa: BLE001
            pass

    bbox_min = mesh.vertices.min(axis=0).tolist() if len(mesh.vertices) else [0, 0, 0]
    bbox_max = mesh.vertices.max(axis=0).tolist() if len(mesh.vertices) else [0, 0, 0]

    return jsonify({
        'status': 'OK',
        'quality': mesh.quality,
        'part': part,
        'downsample_factor': mesh.downsample_factor,
        'vertex_count': int(mesh.vertices.shape[0]),
        'triangle_count': int(mesh.faces.shape[0]),
        'bounding_box_mm': {'min': bbox_min, 'max': bbox_max},
        'warnings': mesh.warnings,
        'segmentation_method': segmentation.method,
        'segmentation_method_version': segmentation.method_version,
        'vertices_b64': encode_typed_array(mesh.vertices, np.float32),
        'faces_b64': encode_typed_array(mesh.faces, np.uint32),
        'normals_b64': encode_typed_array(mesh.normals, np.float32),
    })


@app.route('/api/dicom/study/<study_id>/analysis', methods=['GET'])
@authmod.api_login_required
def get_study_analysis(study_id):
    """
    Structured quantitative analysis of the study: scan quality, lung volumes,
    HU statistics, density histograms, and regional distribution.

    Computed once and cached on the study, because the measurements depend
    only on the volume and the segmentation - not on anything the viewer does.
    Pass ?refresh=1 to recompute.

    Sections that would require capabilities this application does not have
    (lobes, findings, airways, vessels, texture radiomics, prior-study
    comparison) are returned with NOT_AVAILABLE and a reason, never a value.
    """
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err

    if not request.args.get('refresh') and study.get('analysis') is not None:
        return jsonify({'status': 'OK', 'cached': True, 'analysis': study['analysis']})

    job_id = None
    try:
        job_id = measmod.start_job(study_id, "QUANTITATIVE_ANALYSIS")
    except Exception:  # noqa: BLE001
        pass

    try:
        analysis = analyze_study(
            hu_volume=study['hu_volume'],
            hu_available_per_slice=study['hu_available_per_slice'],
            geometry=study['geometry'],
            segmentation=study.get('segmentation'),
            summary=study['summary'],
            series_status=study['summary'].get('series_status', 'PASS'),
        )
    except Exception as exc:  # noqa: BLE001
        if job_id:
            try:
                measmod.fail_job(job_id, str(exc))
            except Exception:  # noqa: BLE001
                pass
        return jsonify({'status': 'FAIL',
                         'message': f'Quantitative analysis failed: {exc}'}), 500

    with STUDIES_LOCK:
        study['analysis'] = analysis
        try:
            STUDIES.save_analysis(study_id, analysis)
        except KeyError:
            pass

    if job_id:
        try:
            measmod.complete_job(job_id)
            measmod.update_study_status(study_id, "analyzed")
        except Exception:  # noqa: BLE001
            pass

    dbmod.record_audit('analysis_generated', doctor_id=authmod.current_doctor()['id'],
                        target_type='study', target_id=study_id, ip=authmod.client_ip())
    return jsonify({'status': 'OK', 'cached': False, 'analysis': analysis})


@app.route('/api/dicom/study/<study_id>/band-profile', methods=['GET'])
@authmod.api_login_required
def get_band_profile(study_id):
    """Density-band composition slice by slice, inferior to superior.

    Answers "where in the lung does this density concentrate" as a curve,
    rather than requiring the reader to scroll and estimate. Cached with the
    study because it depends only on the volume and the segmentation.
    """
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err

    seg = study.get('segmentation')
    if seg is None or not getattr(seg, 'success', False) or seg.mask is None:
        return jsonify({'status': 'NOT_AVAILABLE',
                         'reason': 'Run lung segmentation for this study first.'}), 409

    if not request.args.get('refresh') and study.get('band_profile') is not None:
        return jsonify({'status': 'OK', 'cached': True, 'profile': study['band_profile']})

    profile = band_profile(study['hu_volume'], seg.mask, study['geometry'])
    with STUDIES_LOCK:
        study['band_profile'] = profile
    return jsonify({'status': 'OK', 'cached': False, 'profile': profile})


# ---------------------------------------------------------------------------
# MEASUREMENTS, ANNOTATIONS, REGIONS OF INTEREST
# ---------------------------------------------------------------------------
# Persisted via api/measurements.py against the imaging-relational tables
# (api/models.py). Geometry is always computed by the SERVER from the
# study's own VolumeGeometry affine, never trusted from the client as
# millimetre coordinates (master spec section 31) - the client sends voxel
# indices in the same (x=column, y=row, z=slice) convention as /voxel above,
# and the server resolves those to patient-space mm and, for point_hu, to
# the actual HU value.
#
# Only 'point_hu' and 'distance' are computable today. 'longest_diameter',
# 'perpendicular_diameter', 'area', and 'volume' are valid values in the
# Measurement schema (a clinician could conceivably record one from an
# external tool later) but this endpoint does not fabricate a caliper/area
# algorithm that doesn't exist - see ARCHITECTURE_AUDIT.md.
# ---------------------------------------------------------------------------

def _voxel_to_world_and_hu(study, x, y, z):
    hu_volume = study['hu_volume']
    n_slices, rows, cols = hu_volume.shape
    if not (0 <= x < cols and 0 <= y < rows and 0 <= z < n_slices):
        raise ValueError('Voxel coordinates out of range.')
    world = study['geometry'].to_world(z, y, x)
    hu = float(hu_volume[z, y, x]) if study['hu_available_per_slice'][z] else None
    return world, hu


@app.route('/api/dicom/study/<study_id>/measurements', methods=['POST'])
@authmod.api_login_required
def create_measurement_endpoint(study_id):
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err
    doctor = authmod.current_doctor()
    data = request.get_json(silent=True) or {}
    mtype = data.get('measurement_type')
    voxels = data.get('voxels')

    if mtype not in ('point_hu', 'distance'):
        return jsonify({'status': 'FAIL', 'message':
                         "measurement_type must be 'point_hu' or 'distance'."}), 400
    if not isinstance(voxels, list) or not voxels:
        return jsonify({'status': 'FAIL', 'message':
                         'voxels must be a non-empty list of [x, y, z] integer indices.'}), 400
    expected = 1 if mtype == 'point_hu' else 2
    if len(voxels) != expected:
        return jsonify({'status': 'FAIL', 'message':
                         f"{mtype} requires exactly {expected} voxel(s)."}), 400

    world_points, hu_values = [], []
    for v in voxels:
        try:
            x, y, z = int(v[0]), int(v[1]), int(v[2])
            world, hu = _voxel_to_world_and_hu(study, x, y, z)
        except (TypeError, IndexError, ValueError) as exc:
            return jsonify({'status': 'FAIL', 'message': str(exc) or
                             'Each voxel must be [x, y, z] integers within the volume.'}), 400
        world_points.append(world)
        hu_values.append(hu)

    if mtype == 'point_hu':
        if hu_values[0] is None:
            return jsonify({'status': 'FAIL', 'message':
                             'HU is not available for this voxel (no rescale slope/intercept '
                             'on that slice).'}), 409
        value, units = hu_values[0], 'HU'
    else:
        p0, p1 = np.array(world_points[0]), np.array(world_points[1])
        value, units = float(np.linalg.norm(p1 - p0)), 'mm'

    try:
        measurement_id = measmod.create_measurement(
            study_id, mtype, geometry_mm=[list(p) for p in world_points],
            value=value, units=units, created_by_doctor_id=doctor['id'],
            mean_hu=hu_values[0] if mtype == 'point_hu' else None,
            provenance={'source': 'clinician_measurement', 'voxel_indices': voxels,
                        'transform': 'VolumeGeometry.to_world'},
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({'status': 'FAIL', 'message': f'Could not save measurement: {exc}'}), 500

    dbmod.record_audit('measurement_created', doctor_id=doctor['id'],
                        target_type='study', target_id=study_id, ip=authmod.client_ip())
    return jsonify({'status': 'OK', 'measurement_id': measurement_id,
                     'value': value, 'units': units}), 201


@app.route('/api/dicom/study/<study_id>/measurements', methods=['GET'])
@authmod.api_login_required
def list_measurements_endpoint(study_id):
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err
    try:
        rows = measmod.list_measurements(study_id)
    except Exception as exc:  # noqa: BLE001
        return jsonify({'status': 'FAIL', 'message': str(exc)}), 500
    return jsonify({'status': 'OK', 'measurements': rows})


@app.route('/api/dicom/study/<study_id>/measurements/<measurement_id>', methods=['DELETE'])
@authmod.api_login_required
def delete_measurement_endpoint(study_id, measurement_id):
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err
    doctor = authmod.current_doctor()
    try:
        measmod.delete_measurement(measurement_id, created_by_doctor_id=doctor['id'])
    except KeyError:
        return jsonify({'status': 'NOT_FOUND'}), 404
    except PermissionError:
        return jsonify({'status': 'FORBIDDEN'}), 403
    dbmod.record_audit('measurement_deleted', doctor_id=doctor['id'],
                        target_type='study', target_id=study_id, ip=authmod.client_ip())
    return jsonify({'status': 'OK'})


@app.route('/api/dicom/study/<study_id>/annotations', methods=['POST'])
@authmod.api_login_required
def create_annotation_endpoint(study_id):
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err
    doctor = authmod.current_doctor()
    data = request.get_json(silent=True) or {}
    text = data.get('text')
    voxel = data.get('voxel')  # optional [x, y, z]

    position_mm = None
    if voxel is not None:
        try:
            x, y, z = int(voxel[0]), int(voxel[1]), int(voxel[2])
            position_mm, _hu = _voxel_to_world_and_hu(study, x, y, z)
            position_mm = list(position_mm)
        except (TypeError, IndexError, ValueError) as exc:
            return jsonify({'status': 'FAIL', 'message': str(exc) or
                             'voxel must be [x, y, z] integers within the volume.'}), 400

    try:
        annotation_id = measmod.create_annotation(
            study_id, text, created_by_doctor_id=doctor['id'], position_mm=position_mm)
    except ValueError as exc:
        return jsonify({'status': 'FAIL', 'message': str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({'status': 'FAIL', 'message': f'Could not save annotation: {exc}'}), 500

    dbmod.record_audit('annotation_created', doctor_id=doctor['id'],
                        target_type='study', target_id=study_id, ip=authmod.client_ip())
    return jsonify({'status': 'OK', 'annotation_id': annotation_id}), 201


@app.route('/api/dicom/study/<study_id>/annotations', methods=['GET'])
@authmod.api_login_required
def list_annotations_endpoint(study_id):
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err
    try:
        rows = measmod.list_annotations(study_id)
    except Exception as exc:  # noqa: BLE001
        return jsonify({'status': 'FAIL', 'message': str(exc)}), 500
    return jsonify({'status': 'OK', 'annotations': rows})


@app.route('/api/dicom/study/<study_id>/annotations/<annotation_id>', methods=['DELETE'])
@authmod.api_login_required
def delete_annotation_endpoint(study_id, annotation_id):
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err
    doctor = authmod.current_doctor()
    try:
        measmod.delete_annotation(annotation_id, created_by_doctor_id=doctor['id'])
    except KeyError:
        return jsonify({'status': 'NOT_FOUND'}), 404
    except PermissionError:
        return jsonify({'status': 'FORBIDDEN'}), 403
    dbmod.record_audit('annotation_deleted', doctor_id=doctor['id'],
                        target_type='study', target_id=study_id, ip=authmod.client_ip())
    return jsonify({'status': 'OK'})


@app.route('/api/dicom/study/<study_id>/regions/sync-deterministic', methods=['POST'])
@authmod.api_login_required
def sync_deterministic_regions_endpoint(study_id):
    """Persists the locatable regions already computed by density_regions.py
    (via /analysis) as RegionOfInterest rows, so they survive independently
    of the in-memory/cached analysis blob and can be listed/joined
    relationally. This performs no new computation - it transcribes an
    existing result."""
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err
    analysis = study.get('analysis')
    densitometry = (analysis or {}).get('densitometry')
    if not densitometry or densitometry.get('status') != 'OK':
        return jsonify({'status': 'NOT_AVAILABLE', 'reason':
                         'Run /analysis for this study first; densitometry is not available.'}), 409
    created = 0
    try:
        for band_key, band_result in densitometry.get('regions', {}).items():
            for region in band_result.get('regions', []):
                measmod.create_region(
                    study_id, source='deterministic_segmentation',
                    centroid_mm=tuple(region['centroid_mm']),
                    bbox=region['bounding_box_voxel'],
                    volume_ml=region['volume_ml'], mean_hu=region['mean_hu'],
                    median_hu=region['median_hu'], laterality=region.get('side'),
                    zone=region.get('zone'),
                    provenance={'source': 'density_regions', 'band': band_key,
                                'region_id': region['region_id'],
                                'densitometry_version': densitometry.get('densitometry_version')},
                )
                created += 1
    except Exception as exc:  # noqa: BLE001
        return jsonify({'status': 'FAIL', 'message': f'Could not persist regions: {exc}'}), 500
    return jsonify({'status': 'OK', 'regions_created': created})


@app.route('/api/dicom/study/<study_id>/regions', methods=['GET'])
@authmod.api_login_required
def list_regions_endpoint(study_id):
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err
    source = request.args.get('source')
    if source and source not in ('clinician_annotation', 'deterministic_segmentation'):
        return jsonify({'status': 'FAIL', 'message':
                         "source must be 'clinician_annotation' or 'deterministic_segmentation'."}), 400
    try:
        rows = measmod.list_regions(study_id, source=source)
    except Exception as exc:  # noqa: BLE001
        return jsonify({'status': 'FAIL', 'message': str(exc)}), 500
    return jsonify({'status': 'OK', 'regions': rows})


@app.route('/api/dicom/study/<study_id>/jobs', methods=['GET'])
@authmod.api_login_required
def list_jobs_endpoint(study_id):
    """Processing-job history for this study. See models.ProcessingJob's
    docstring: every job here ran synchronously within its triggering
    request (no async worker exists on this deployment yet) - QUEUED is
    never observed, only RUNNING -> COMPLETED/FAILED."""
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err
    try:
        rows = measmod.list_jobs(study_id)
    except Exception as exc:  # noqa: BLE001
        return jsonify({'status': 'FAIL', 'message': str(exc)}), 500
    return jsonify({'status': 'OK', 'jobs': rows})


@app.route('/api/dicom/study/<study_id>/volume-texture', methods=['GET'])
@authmod.api_login_required
def get_volume_texture(study_id):
    """
    Serves the real reconstructed HU volume (downsampled only if it exceeds
    the requested/max dimension) for client-side direct volume rendering.
    The client applies its own window/level transfer function in the GPU
    shader, so no windowing is baked in server-side here.
    """
    study, err = _get_owned_study_or_error(study_id)
    if err:
        return err

    try:
        max_dim = int(request.args.get('max_dim', VOLUME_TEXTURE_DEFAULT_MAX_DIM))
    except ValueError:
        return jsonify({'status': 'FAIL', 'message': 'max_dim must be an integer.'}), 400
    max_dim = max(16, min(max_dim, VOLUME_TEXTURE_HARD_CAP_DIM))

    hu_volume = study['hu_volume']
    n_slices, rows, cols = hu_volume.shape
    native_max = max(n_slices, rows, cols)
    downsample_factor = max(1, int(np.ceil(native_max / max_dim)))

    if downsample_factor > 1:
        sampled = hu_volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
        warning = (
            f'Volume texture downsampled {downsample_factor}x from native resolution '
            f'({n_slices}x{rows}x{cols}) to fit within a {max_dim}-voxel-per-axis browser '
            f'texture cap; the high-fidelity surface mesh is unaffected by this limit.'
        )
    else:
        sampled = hu_volume
        warning = None

    spacing = study['geometry'].spacing_mm  # (x, y, z) mm per NATIVE voxel
    effective_spacing = [s * downsample_factor for s in spacing]

    clipped = np.clip(sampled, -1024, 3071).astype(np.int16)
    hu_available_fraction = sum(1 for v in study['hu_available_per_slice'] if v) / n_slices

    return jsonify({
        'status': 'OK',
        'shape': {'slices': clipped.shape[0], 'rows': clipped.shape[1], 'cols': clipped.shape[2]},
        'spacing_mm': effective_spacing,
        'downsample_factor': downsample_factor,
        'hu_available_fraction': round(hu_available_fraction, 4),
        'warning': warning,
        'data_b64': encode_typed_array(clipped, np.int16),
    })


# ---------------------------------------------------------------------------
# CASES & NOTES API
# ---------------------------------------------------------------------------
# Every endpoint below authenticates, then re-verifies case authorization
# server-side via authmod.authorize_case_or_none. Denied and non-existent
# cases return an identical 404 so case refs cannot be enumerated.
# ---------------------------------------------------------------------------

@app.route('/api/cases', methods=['GET'])
@authmod.api_login_required
def api_list_cases():
    doctor = authmod.current_doctor()
    rows = casemod.list_cases_for_doctor(
        doctor['id'],
        status=request.args.get('status'),
        query=request.args.get('q'),
    )
    return jsonify({'status': 'OK', 'cases': [casemod.case_to_dict(r) for r in rows]})


@app.route('/api/cases', methods=['POST'])
@authmod.api_login_required
def api_create_case():
    doctor = authmod.current_doctor()
    payload = request.get_json(silent=True) or {}
    title = (payload.get('title') or '').strip()
    if not title:
        return jsonify({'status': 'FAIL', 'message': 'A case title is required.'}), 400
    row = casemod.create_case(doctor['id'], title,
                               status=payload.get('status', 'needs_review'),
                               is_demo=bool(payload.get('is_demo')))
    dbmod.record_audit('case_created', doctor_id=doctor['id'],
                        target_type='case', target_id=row['id'], ip=authmod.client_ip())
    return jsonify({'status': 'OK', 'case': casemod.case_to_dict(row)}), 201


@app.route('/api/cases/<case_ref>', methods=['GET'])
@authmod.api_login_required
def api_get_case(case_ref):
    doctor = authmod.current_doctor()
    case_row, _reason = authmod.authorize_case_or_none(case_ref, doctor['id'])
    if case_row is None:
        dbmod.record_audit('case_access_denied', doctor_id=doctor['id'],
                            target_type='case_ref', target_id=case_ref,
                            outcome='denied', ip=authmod.client_ip())
        return jsonify({'status': 'NOT_FOUND', 'message': 'Case not found.'}), 404
    notes = casemod.list_notes(case_row['id'])
    return jsonify({'status': 'OK', 'case': casemod.case_to_dict(case_row),
                     'notes': [casemod.note_to_dict(n) for n in notes]})


@app.route('/api/cases/<case_ref>/notes', methods=['POST'])
@authmod.api_login_required
def api_add_note(case_ref):
    doctor = authmod.current_doctor()
    case_row, _reason = authmod.authorize_case_or_none(case_ref, doctor['id'])
    if case_row is None:
        dbmod.record_audit('case_access_denied', doctor_id=doctor['id'],
                            target_type='case_ref', target_id=case_ref,
                            outcome='denied', ip=authmod.client_ip())
        return jsonify({'status': 'NOT_FOUND', 'message': 'Case not found.'}), 404
    payload = request.get_json(silent=True) or {}
    try:
        note_id = casemod.add_note(case_row['id'], doctor['id'], payload.get('content'))
    except ValueError as exc:
        return jsonify({'status': 'FAIL', 'message': str(exc)}), 400
    # Audit records the note id only - never the note text.
    dbmod.record_audit('note_created', doctor_id=doctor['id'],
                        target_type='note', target_id=note_id, ip=authmod.client_ip())
    return jsonify({'status': 'OK', 'note_id': note_id}), 201


@app.route('/api/cases/<case_ref>/status', methods=['POST'])
@authmod.api_login_required
def api_update_case_status(case_ref):
    doctor = authmod.current_doctor()
    case_row, _reason = authmod.authorize_case_or_none(case_ref, doctor['id'])
    if case_row is None:
        return jsonify({'status': 'NOT_FOUND', 'message': 'Case not found.'}), 404
    payload = request.get_json(silent=True) or {}
    try:
        casemod.update_case_status(case_row['id'], payload.get('status'))
    except ValueError as exc:
        return jsonify({'status': 'FAIL', 'message': str(exc)}), 400
    dbmod.record_audit('case_status_changed', doctor_id=doctor['id'],
                        target_type='case', target_id=case_row['id'], ip=authmod.client_ip())
    return jsonify({'status': 'OK'})


@app.route('/api/session', methods=['GET'])
def api_session():
    """Lets the frontend discover whether a session is active. Returns only
    non-sensitive identity fields."""
    doctor = authmod.current_doctor()
    if doctor is None:
        return jsonify({'authenticated': False}), 200
    return jsonify({'authenticated': True, 'display_name': doctor['display_name'],
                     'email': doctor['email']}), 200


# Required for Vercel deployment
if __name__ == '__main__':
    # Port is configurable because macOS ControlCenter (AirPlay Receiver)
    # occupies port 5000 by default: PORT=5050 python api/index.py
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
