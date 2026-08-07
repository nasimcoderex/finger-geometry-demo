// Port of main.py's draw_banner logic, as an HTML overlay instead of
// cv2.putText - keyed off the same status strings ring.py/ringPose.js
// already produce, so no new state machine is needed here.
const MESSAGES = {
  not_straight: "Straighten your ring finger to place the ring",
  fingers_together: "Spread your fingers apart to place the ring",
};

export function updateBanner(bannerEl, ringStatus) {
  const text = MESSAGES[ringStatus];
  if (text) {
    bannerEl.textContent = text;
    bannerEl.style.display = "block";
  } else {
    bannerEl.style.display = "none";
  }
}
