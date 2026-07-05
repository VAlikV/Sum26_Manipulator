import cv2
import numpy as np


def _as_uint8_image(image):
    if image is None:
        raise ValueError("image is None")

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    return np.clip(image, 0, 255).astype(np.uint8)


def _to_rgb(image, input_format):
    image = _as_uint8_image(image)

    if input_format == "rgb":
        return image
    if input_format == "bgr":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    raise ValueError("input_format must be 'rgb' or 'bgr'")


def make_red_mask(
    image,
    input_format="rgb",
    saturation_min=40,
    value_min=25,
    red_min=45,
    dominance=1.25,
    kernel_size=3,
):
    rgb = _to_rgb(image, input_format)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower_red_1 = np.array([0, saturation_min, value_min], dtype=np.uint8)
    upper_red_1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([168, saturation_min, value_min], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

    hsv_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    hsv_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    hsv_mask = cv2.bitwise_or(hsv_mask_1, hsv_mask_2)

    rgb_int = rgb.astype(np.int16)
    red = rgb_int[:, :, 0]
    green = rgb_int[:, :, 1]
    blue = rgb_int[:, :, 2]
    dominant_red = (
        (red >= red_min)
        & (red >= dominance * (green + 1))
        & (red >= dominance * (blue + 1))
    )
    rgb_mask = dominant_red.astype(np.uint8) * 255

    mask = cv2.bitwise_or(hsv_mask, rgb_mask)

    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def detect_red_cube(image, min_area=200, input_format="rgb"):
    rgb = _to_rgb(image, input_format)
    mask = make_red_mask(rgb, input_format="rgb")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_pixels = int(np.count_nonzero(mask))

    result = {
        "detected": False,
        "bbox": None,
        "center": None,
        "area": 0.0,
        "mask": mask,
        "mask_pixels": mask_pixels,
        "max_red": int(rgb[:, :, 0].max()),
    }

    if not contours:
        return result

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    result["area"] = area

    if area < min_area:
        return result

    x, y, w, h = cv2.boundingRect(contour)
    result.update(
        {
            "detected": True,
            "bbox": np.array([x, y, w, h], dtype=np.int64),
            "center": np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float64),
        }
    )
    return result


def draw_detection(image, detection, input_format="rgb"):
    annotated = _as_uint8_image(image).copy()
    if detection is None or not detection.get("detected", False):
        return annotated

    x, y, w, h = detection["bbox"]
    cx, cy = detection["center"].astype(int)

    if input_format not in ("rgb", "bgr"):
        raise ValueError("input_format must be 'rgb' or 'bgr'")

    color = (0, 255, 0)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    cv2.circle(annotated, (cx, cy), 4, color, -1)
    return annotated


def detect_red_cube_vis(image, min_area=200, input_format="rgb"):
    detection = detect_red_cube(image, min_area=min_area, input_format=input_format)
    annotated = draw_detection(image, detection, input_format=input_format)

    if input_format == "rgb":
        source_bgr = cv2.cvtColor(_as_uint8_image(image), cv2.COLOR_RGB2BGR)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    elif input_format == "bgr":
        source_bgr = _as_uint8_image(image)
        annotated_bgr = annotated
    else:
        raise ValueError("input_format must be 'rgb' or 'bgr'")

    cv2.imshow("Image", source_bgr)
    cv2.imshow("Red mask", detection["mask"])
    cv2.imshow("Detection", annotated_bgr)
    print(
        "detected={detected} area={area:.1f} mask_pixels={mask_pixels} max_red={max_red}".format(
            **detection
        )
    )
    return detection


if __name__ == "__main__":
    test_image = cv2.imread("A.jpg")
    if test_image is None:
        raise FileNotFoundError("Cannot read A.jpg")

    detect_red_cube_vis(test_image, min_area=40, input_format="bgr")
    cv2.waitKey(0)
