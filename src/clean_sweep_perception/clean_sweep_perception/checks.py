"""Physical sanity checks that reject false positives the detector can't rule out itself."""

from __future__ import annotations

from clean_sweep_perception.geometry import Localization

# Generous upper bounds (width, height) in meters for each class.
SIZE_LIMITS = {
    "bottle": (0.15, 0.40),
    "can": (0.15, 0.25),
    "cup": (0.15, 0.25),
    "paper": (0.40, 0.25),
    "other": (0.40, 0.35),
}
MIN_WIDTH = 0.015  # m; anything narrower is noise or too small to grasp
ON_FLOOR_TOLERANCE = 0.05  # m; the object's lowest visible point must be this close to the floor


def rejection_reason(localization: Localization, label: str | None,
                     top_down: bool = False) -> str | None:
    """Why this detection can't be litter on the floor, or None if it passes.

    `label` may be None when the type is unknown; the loosest limits apply then.
    `top_down`: the camera looks straight down (the wrist camera), so it only sees an
    object's top. Its lowest visible point is then the top, and an object held in the
    gripper is high up: skip the on-floor check and size-check only the visible height.
    """
    if not top_down and localization.bottom > ON_FLOOR_TOLERANCE:
        return f"not on the floor (lowest point {localization.bottom:.2f} m up)"
    if localization.width < MIN_WIDTH:
        return f"too small ({localization.width * 100:.1f} cm wide)"
    max_width, max_height = SIZE_LIMITS.get(label or "other", SIZE_LIMITS["other"])
    height = localization.height - localization.bottom if top_down else localization.height
    if localization.width > max_width or height > max_height:
        return (f"too big for {label or 'litter'} "
                f"({localization.width * 100:.0f} x {localization.height * 100:.0f} cm)")
    return None
