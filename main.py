import mss
import cv2
import numpy as np

ICON_PATH = "quest_icon.png"

COMPASS_ROI = {
    "top": 200,
    "left": 400,
    "width": 1000,
    "height": 150
}

THRESHOLD = 0.6
# ---------------------

def load_template():
    template = cv2.imread(ICON_PATH, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"ERROR: Could not find '{ICON_PATH}'")
        exit()
    return template

def find_marker(frame, template):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(
        gray_frame,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(f"Match confidence: {max_val:.2f}", flush=True)

    if max_val >= THRESHOLD:
        h, w = template.shape[:2]
        center_x = max_loc[0] + w // 2

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

        return center_x

    return None

def start_engine():
    quest_icon = load_template()
    print("Template loaded. Scanner running... (press Q to quit)")

    with mss.mss() as sct:
        while True:
            screenshot = sct.grab(COMPASS_ROI)
            img = np.array(screenshot)

            frame = np.ascontiguousarray(img[:, :, :3])

            marker_x = find_marker(frame, quest_icon)

            if marker_x is not None:
                roi_center = COMPASS_ROI["width"] // 2

                if marker_x < roi_center - 50:
                    status = "<< TURN LEFT"
                elif marker_x > roi_center + 50:
                    status = "TURN RIGHT >>"
                else:
                    status = "^ FORWARD ^"

                cv2.putText(
                    frame,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Assassin's Assistant Debug", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_engine()
