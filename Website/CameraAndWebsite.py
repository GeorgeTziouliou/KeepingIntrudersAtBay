"""
Main application file for the Smart Home Security System.
Refactored to separate processing/decision logic from annotation drawing
for independent stream annotation control.
"""

import cv2
import dlib
import pickle
import numpy as np
from scipy.spatial import distance
import time
from collections import defaultdict
from flask import Flask, Response, render_template, jsonify, send_from_directory, request
import threading
import os
from rfid_routes import add_rfid_routes
from door_control import initialize_door_controller

# --- Configuration ---
CONSECUTIVE_FRAMES_THRESHOLD = 5
USE_BLINK_DETECTION = True
DEBUG_FORCE_LIVENESS_TRUE = False # Keep for debugging if needed
# --------------------

app = Flask(__name__, static_folder='.', template_folder='.')

try:
    from lcd_controller import lcd_controller
    LCD_AVAILABLE = True
except ImportError:
    print("LCD controller not available, running in LCD-less mode")
    LCD_AVAILABLE = False

door_controller = initialize_door_controller(lcd_controller if LCD_AVAILABLE else None)

system_settings = { "use_liveness": True, "allow_other": False }

MODEL_PATH = "face_recognizer_model.pkl"
LABELS_PATH = "label_names.pkl"

# --- Load Models ---
try:
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
    with open(LABELS_PATH, "rb") as f: label_dict = pickle.load(f)
    print("Models loaded successfully.")
except Exception as e: print(f"FATAL ERROR loading models: {e}. Check paths."); exit()
# --------------------

latest_match = { "match_found": False, "person_name": None, "confidence": 0, "timestamp": 0 }
lock = threading.Lock()

# --- Face State & Liveness Detector Classes---
class FaceLivenessState:
    """Stores state for a single detected face over time."""
    def __init__(self):
        self.motion_history = []
        self.last_seen = time.time()
        self.total_blinks = 0
        self.last_blink_frame_time = 0
        self.consecutive_success_count = 0

class LivenessDetector:
    """Manages liveness detection states and checks."""
    def __init__(self):
        self.face_states = defaultdict(FaceLivenessState)
        self.cleanup_interval = 5; self.last_cleanup = time.time()

    def cleanup_old_faces(self):
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            inactive = [fid for fid, state in self.face_states.items() if current_time - state.last_seen > self.cleanup_interval]
            for fid in inactive: self.face_states.pop(fid, None)
            self.last_cleanup = current_time

    def get_face_id(self, face, fw):
        cx = (face.left() + face.right()) / 2; cy = (face.top() + face.bottom()) / 2
        return f"{int(cx/fw*100)}_{int(cy/fw*100)}" if fw > 0 else "0_0"

    def get_eye_aspect_ratio(self, eye):
        try: A=distance.euclidean(eye[1],eye[5]); B=distance.euclidean(eye[2],eye[4]); C=distance.euclidean(eye[0],eye[3]); return (A+B)/(2.0*C) if C>1e-3 else 1.0
        except: return 1.0

    def detect_blink(self, shape, state):
        try:
            if not USE_BLINK_DETECTION: return True
            if shape.num_parts < 48: return False
            le = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
            re = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])
            ear = (self.get_eye_aspect_ratio(le) + self.get_eye_aspect_ratio(re)) / 2.0
            ct = time.time()
            if ear < 0.3:
                if ct - state.last_blink_frame_time > 0.4: state.total_blinks += 1
                state.last_blink_frame_time = ct
            # else: state.last_blink_frame_time = 0 # Reset debounce only if needed
            return state.total_blinks > 0
        except Exception as e: print(f"Blink Error: {e}"); return False

    def detect_movement(self, face, frame_shape, state):
        try:
            if len(frame_shape)<2 or frame_shape[0]==0 or frame_shape[1]==0: return False
            cx=(face.left()+face.right())/2; cy=(face.top()+face.bottom())/2
            nx=cx/frame_shape[1]; ny=cy/frame_shape[0]
            state.motion_history.append((nx, ny))
            if len(state.motion_history) > 30: state.motion_history.pop(0)
            if len(state.motion_history) >= 10:
                xs=[x for x,_ in state.motion_history]; ys=[y for _,y in state.motion_history]
                if not xs or not ys: return False
                var = np.var(xs) + np.var(ys); return 0.00003 < var < 0.05 # WIDER range test
            else: return True # Default to TRUE until enough history
        except Exception as e: print(f"Movement Error: {e}"); return False

    def check_texture_variance(self, frame, face):
        try:
            x1,y1=max(0,face.left()),max(0,face.top()); x2,y2=min(face.right(),frame.shape[1]),min(face.bottom(),frame.shape[0])
            wc,hc = x2-x1, y2-y1;
            if wc<=0 or hc<=0: return False
            roi=frame[y1:y2, x1:x2];
            if roi.size==0: return False
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY); var = np.var(gray_roi)
            return var > 300 # LOWER threshold test
        except Exception as e: print(f"Texture Error: {e}"); return False

# --- Refactored Core Processing Function ---
def process_single_frame(frame, liveness_detector):
    """
    Performs detection, recognition, liveness checks, and MFA logic for all faces.
    Does NOT draw annotations. Returns state data for each face and unlock trigger info.
    """
    global door_controller, latest_match, system_settings, CONSECUTIVE_FRAMES_THRESHOLD, lock

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    liveness_detector.cleanup_old_faces()

    face_data_list = [] # Store results for each face
    match_should_trigger_unlock_sustained = False
    unlock_info = None

    liveness_globally_enabled = system_settings.get("use_liveness", True) # Check global setting once

    for face in faces:
        face_id = liveness_detector.get_face_id(face, frame.shape[1])
        face_state = liveness_detector.face_states[face_id]
        face_state.last_seen = time.time()

        # --- Recognition ---
        try:
            shape = sp(gray, face)
            descriptor = facerec.compute_face_descriptor(frame, shape)
            descriptor = np.array(descriptor).reshape(1, -1)
            pred = model.predict(descriptor)[0]
            dist, _ = model.kneighbors(descriptor)
            conf = max(0, min(100, 100 - (dist[0][0] * 100)))
            name = "Other" if conf < 50 else label_dict.get(pred, "Unknown")
        except Exception as e:
            print(f"Recog Error: {e}"); name = "Error"; conf = 0; shape=None # Set shape none on error

        # --- Liveness Calculation (only if globally enabled) ---
        is_live = True # Default
        if liveness_globally_enabled and shape is not None: # Need shape for checks
            if DEBUG_FORCE_LIVENESS_TRUE:
                is_live = True
            else:
                blink_ok = liveness_detector.detect_blink(shape, face_state)
                move_ok = liveness_detector.detect_movement(face, frame.shape, face_state)
                tex_ok = liveness_detector.check_texture_variance(frame, face)
                is_live = move_ok and tex_ok
                if USE_BLINK_DETECTION: is_live = is_live and blink_ok

        # --- MFA Logic Check ---
        meets_criteria = False
        rfid_matched = False
        current_rfid = None
        if name not in ["Unknown", "Error"] and door_controller and door_controller.is_locked():
            current_rfid = door_controller.get_rfid_name()
            live_ok_for_unlock = not liveness_globally_enabled or is_live
            if current_rfid and len(str(current_rfid).strip()) > 0:
                rfid_matched = door_controller.check_face_match(name)
                if rfid_matched and live_ok_for_unlock:
                    meets_criteria = (name != "Other" or system_settings.get("allow_other", False))

        # --- Update Streak & Check Threshold ---
        if meets_criteria: face_state.consecutive_success_count += 1
        else: face_state.consecutive_success_count = 0 # Reset on any failure

        if face_state.consecutive_success_count >= CONSECUTIVE_FRAMES_THRESHOLD:
            if not match_should_trigger_unlock_sustained: # Only trigger once per frame cycle
                match_should_trigger_unlock_sustained = True
                unlock_info = (name, conf)
                # Update global notification state
                ctime = time.time()
                with lock:
                    if ctime - latest_match["timestamp"] > 5 or latest_match["person_name"] != name:
                        latest_match.update({"match_found": True, "person_name": name, "confidence": conf, "timestamp": ctime})

        # Store results for this face
        face_data_list.append({
            "box": face,
            "name": name,
            "confidence": conf,
            "is_live": is_live, # The calculated liveness state
            "rfid_checked": current_rfid is not None,
            "rfid_matched": rfid_matched,
            "current_rfid_data": current_rfid,
            "streak_count": face_state.consecutive_success_count,
            "meets_criteria_this_frame": meets_criteria # Instantaneous check result
        })

    # --- Actual Unlock Trigger Logic---
    if match_should_trigger_unlock_sustained and door_controller and door_controller.is_locked():
        print(f"-------> UNLOCK TRIGGERED at time: {time.time()} (Sustained Check Passed)")
        p_name, p_conf = unlock_info
        print(f"System: Sustained match! Authorizing access for {p_name} ({p_conf:.1f}%)")
        threading.Thread(target=door_controller.unlock_door, daemon=True).start()
        # Return flag indicating trigger happened this cycle
        unlock_triggered_this_cycle = True
    else:
        unlock_triggered_this_cycle = False

    return face_data_list, unlock_triggered_this_cycle, unlock_info


# --- Annotation Function ---
def draw_annotations(frame, face_data_list, show_liveness_annotations=True):
    """Draws annotations onto the frame based on processed face data."""
    global system_settings, CONSECUTIVE_FRAMES_THRESHOLD # Access needed settings

    annotated_frame = frame.copy()
    liveness_globally_enabled = system_settings.get("use_liveness", True)

    for data in face_data_list:
        face = data["box"]
        name = data["name"]
        confidence = data["confidence"]
        is_live = data["is_live"]
        rfid_checked = data["rfid_checked"]
        rfid_matched = data["rfid_matched"]
        current_rfid = data["current_rfid_data"]
        streak_count = data["streak_count"]
        meets_criteria = data["meets_criteria_this_frame"]
        threshold_met = streak_count >= CONSECUTIVE_FRAMES_THRESHOLD

        # --- Determine Annotation Color ---
        color = (128, 128, 128) # Default grey
        if threshold_met: color = (0, 255, 0) # Green if threshold met
        elif meets_criteria: color = (0, 255, 255) # Yellow if valid this frame but threshold not met
        elif show_liveness_annotations and liveness_globally_enabled and not is_live: color = (0, 0, 255) # Red if liveness shown, enabled, and failed
        elif name == "Unknown": color = (255, 0, 0) # Blue if unknown
        elif name == "Error": color = (0, 165, 255) # Orange if error
        # (Grey remains for cases like known face, live, but wrong/no RFID)

        # --- Draw Annotations ---
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated_frame, f"{name}: {confidence:.1f}%", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Liveness text: only if requested for *this* stream AND globally enabled
        if show_liveness_annotations and liveness_globally_enabled:
            cv2.putText(annotated_frame, f"Liveness: {'Live' if is_live else 'Fake'}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # RFID status text
        rfid_text, rfid_color = "No Card", (255, 255, 0)
        if rfid_checked:
            rfid_name = current_rfid if current_rfid and len(str(current_rfid).strip())>0 else "Present"
            rfid_text, rfid_color = ("RFID Match: YES", (0,255,0)) if rfid_matched else (f"RFID Match: NO ({rfid_name})",(0,0,255))
        elif name not in ["Unknown","Error"]: rfid_text, rfid_color = "RFID N/A", (128, 128, 128)
        cv2.putText(annotated_frame, rfid_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rfid_color, 1)

        # Streak text
        cv2.putText(annotated_frame, f"Streak: {streak_count}/{CONSECUTIVE_FRAMES_THRESHOLD}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated_frame

# --- Global variables for camera stream ---
camera = None
output_frame_with_liveness = None
output_frame_without_liveness = None


def generate_frames(frame_type='with_liveness'):
    """Generator for MJPEG streaming."""
    global output_frame_with_liveness, output_frame_without_liveness, lock
    while True:
        time.sleep(0.03) # Stream frame rate target
        frame_to_encode = None
        with lock:
            selected = output_frame_with_liveness if frame_type == 'with_liveness' else output_frame_without_liveness
            if selected is not None: frame_to_encode = selected.copy()
        if frame_to_encode is None: continue
        try:
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if flag: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        except Exception as e: print(f"Encoding error: {e}")


def process_frames():
    """Main frame processing loop - uses new structure."""
    global camera, output_frame_with_liveness, output_frame_without_liveness, lock

    liveness_detector = LivenessDetector()
    print("Starting frame processing thread...")
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Capture frame failed. Restarting..."); time.sleep(1)
            if camera: camera.release()
            camera = cv2.VideoCapture(0)
            if not camera.isOpened(): print("FATAL: Cam restart fail."); break
            else: print("Camera restarted."); continue

        # --- 1. Process frame ONCE to get state data ---
        face_data_list, unlock_triggered, unlock_info = process_single_frame(frame, liveness_detector)

        # --- 2. Create annotated frames based on returned data ---
        frame_copy_live = frame.copy()
        frame_copy_no_live = frame.copy()

        annotated_live = draw_annotations(frame_copy_live, face_data_list, show_liveness_annotations=True)
        annotated_no_live = draw_annotations(frame_copy_no_live, face_data_list, show_liveness_annotations=False)

        # --- 3. Add "Access Granted" overlay ONLY if triggered THIS cycle ---
        if unlock_triggered and unlock_info:
            person_name_trigger, _ = unlock_info
            text=f"Access Granted: {person_name_trigger}";(tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
            # Draw on BOTH frames for consistency
            for f in [annotated_live, annotated_no_live]:
                fh,fw=f.shape[:2];tx,ty=(fw-tw)//2,fh-20
                cv2.rectangle(f,(tx-5,ty-th-5),(tx+tw+5,ty+5),(0,0,0),-1)
                cv2.putText(f,text,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        # --- 4. Update shared buffers ---
        with lock:
            output_frame_with_liveness = annotated_live
            output_frame_without_liveness = annotated_no_live

def match_status_reset():
    """Resets match notification flag periodically."""

    global latest_match, lock
    print("Starting match status reset thread...")
    while True:
        time.sleep(1); current_time = time.time()
        with lock:
            if latest_match["match_found"] and (current_time - latest_match["timestamp"] > 5):
                latest_match["match_found"] = False; print("Match status notification reset")

# --- Flask Routes ---
@app.route('/static/<path:filename>')
def serve_static(filename): return send_from_directory('.', filename)
@app.route('/')
def index(): return render_template('index.html')
@app.route('/video_feed_with_liveness')
def video_feed_with_liveness(): return Response(generate_frames(frame_type='with_liveness'), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed_without_liveness')
def video_feed_without_liveness(): return Response(generate_frames(frame_type='without_liveness'), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/door_status', methods=['GET'])
def get_door_status():
    if door_controller: return jsonify(door_controller.door_status)
    else: return jsonify({"error": "DC unavailable"}), 500
@app.route('/toggle_door/<int:status>', methods=['POST'])
def toggle_door(status):
    if not door_controller: return jsonify({"success": False, "error": "DC unavailable"}), 500
    if status==1: door_controller.lock_door()
    elif status==0: door_controller.unlock_door()
    else: return jsonify({"success": False, "error": "Invalid status"}), 400
    return jsonify({"success": True, "locked": door_controller.is_locked()})
@app.route('/current_rfid_name', methods=['GET'])
def current_rfid_name():
    rfid_name = door_controller.get_rfid_name() if door_controller else None
    return jsonify({"name": rfid_name})
@app.route('/match_status', methods=['GET'])
def match_status():
    global latest_match, lock;
    with lock: d = latest_match.copy()
    return jsonify(d)
@app.route('/system_settings', methods=['GET'])
def get_system_settings():
    global system_settings, lock;
    with lock: d = system_settings.copy()
    return jsonify(d)
@app.route('/update_settings', methods=['POST'])
def update_settings():
    global system_settings, lock
    if not request.is_json: return jsonify({"success": False, "error": "Req JSON"}), 400
    try:
        data = request.get_json(); updated = False
        with lock:
            if 'use_liveness' in data and isinstance(data['use_liveness'], bool):
                if system_settings['use_liveness'] != data['use_liveness']: system_settings['use_liveness'] = data['use_liveness']; updated = True; print(f"Set use_liveness={system_settings['use_liveness']}")
            if 'allow_other' in data and isinstance(data['allow_other'], bool):
                if system_settings['allow_other'] != data['allow_other']: system_settings['allow_other'] = data['allow_other']; updated = True; print(f"Set allow_other={system_settings['allow_other']}")
        return jsonify({"success": True, "message": "Settings updated" if updated else "No changes"})
    except Exception as e: print(f"Update error: {e}"); return jsonify({"success": False, "error":"Server error"}), 500

# --- Main ---
def main():
    """Initializes and starts application components."""
    global camera
    print("Init Cam..."); camera = cv2.VideoCapture(0)
    if not camera.isOpened(): print("FATAL: No webcam."); return
    print("Init RFID..."); rfid_cleanup = add_rfid_routes(app, door_controller)
    if LCD_AVAILABLE:
        print("Init LCD..."); lcd_ok = False
        try:
            if lcd_controller.initialize():
                lcd_controller.start_message_thread()
                lcd_controller.display_message("Smart Home Sec", "Initializing..."); time.sleep(2)
                if door_controller: lcd_controller.update_door_status(door_controller.is_locked())
                print("LCD OK."); lcd_ok = True
        except Exception as e: print(f"LCD setup error: {e}")
        if not lcd_ok: print("Warn: LCD init fail.")
    print("Start threads...");
    threading.Thread(target=process_frames, name="FrameProc", daemon=True).start()
    threading.Thread(target=match_status_reset, name="MatchReset", daemon=True).start()
    print("Start Flask http://0.0.0.0:5000 ...");
    try: app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except Exception as e: print(f"FATAL Flask err: {e}")
    finally:
        # --- Cleanup ---
        print("\nShutting down... Cleaning up resources...")
        # Stop camera
        if camera is not None:
            print("Releasing camera...")
            camera.release()

        # Clean up LCD
        if LCD_AVAILABLE:
            try:
                print("Cleaning up LCD...")
                lcd_controller.cleanup()
            except Exception as e:
                print(f"Error during LCD cleanup: {e}")

        # Clean up Door Controller (e.g., cancel timers)
        # Add check if door_controller exists before cleanup
        if 'door_controller' in globals() and door_controller:
            try:
                print("Cleaning up Door Controller...")
                door_controller.cleanup()
            except Exception as e:
                print(f"Error during Door Controller cleanup: {e}")

        # Clean up RFID resources (stops thread, cleans GPIO)
        # Add check if rfid_cleanup function exists before calling
        if 'rfid_cleanup' in locals() or 'rfid_cleanup' in globals():
            try:
                print("Cleaning up RFID...")
                rfid_cleanup()
            except Exception as e:
                print(f"Error during RFID cleanup: {e}")

        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()