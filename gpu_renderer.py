import json
import os
import subprocess
import sys
import time
from collections import deque

if (
    sys.platform.startswith("linux")
    and os.environ.get("DISPLAY")
    and "GST_GL_PLATFORM" not in os.environ
):
    # ponytail: XWayland provides window decorations; use GTK for native Wayland chrome.
    os.environ["GST_GL_PLATFORM"] = "glx"
    os.environ["GST_GL_WINDOW"] = "x11"
    os.environ["GST_GL_API"] = "opengl"
elif sys.platform == "win32":
    os.environ.setdefault("GST_GL_API", "opengl")

import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstGL", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import GObject, Gst, GstGL, GstVideo

Gst.init(None)

USE_GLES = (
    sys.platform.startswith("linux")
    and bool(os.environ.get("WAYLAND_DISPLAY"))
    and os.environ.get("GST_GL_API") != "opengl"
)
SHADER_VERSION = "#version 100\n" if USE_GLES else "#version 120\n"


VERTEX_SHADER = SHADER_VERSION + r"""attribute vec4 a_position;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;

void main() {
    gl_Position = a_position;
    v_texcoord = a_texcoord;
}
"""


FRAGMENT_SHADER = SHADER_VERSION + r"""#ifdef GL_ES
precision highp float;
#endif

varying vec2 v_texcoord;
uniform sampler2D tex;
uniform float active;
uniform float mode;
uniform float phase;
uniform float frame_w;
uniform float frame_h;
uniform float l0x;
uniform float l0y;
uniform float r0x;
uniform float r0y;
uniform float l1x;
uniform float l1y;
uniform float r1x;
uniform float r1y;
uniform float l2x;
uniform float l2y;
uniform float r2x;
uniform float r2y;
uniform float l3x;
uniform float l3y;
uniform float r3x;
uniform float r3y;
uniform float l4x;
uniform float l4y;
uniform float r4x;
uniform float r4y;

float cross2(vec2 a, vec2 b) {
    return a.x * b.y - a.y * b.x;
}

float inside_triangle(vec2 p, vec2 a, vec2 b, vec2 c) {
    float s0 = cross2(b - a, p - a);
    float s1 = cross2(c - b, p - b);
    float s2 = cross2(a - c, p - c);
    float positive = step(0.0, s0) * step(0.0, s1) * step(0.0, s2);
    float negative = step(s0, 0.0) * step(s1, 0.0) * step(s2, 0.0);
    return max(positive, negative);
}

float inside_quad(vec2 p, vec2 a, vec2 b, vec2 c, vec2 d) {
    return max(inside_triangle(p, a, b, c), inside_triangle(p, a, c, d));
}

float segment_distance(vec2 p, vec2 a, vec2 b) {
    vec2 edge = b - a;
    float along = clamp(dot(p - a, edge) / max(dot(edge, edge), 0.0001), 0.0, 1.0);
    return length(p - (a + along * edge));
}

float quad_border(
    vec2 p,
    vec2 a,
    vec2 b,
    vec2 c,
    vec2 d,
    vec2 frame_size,
    float inside
) {
    vec2 pixel = p * frame_size;
    float distance = min(
        min(segment_distance(pixel, a * frame_size, b * frame_size),
            segment_distance(pixel, b * frame_size, c * frame_size)),
        min(segment_distance(pixel, c * frame_size, d * frame_size),
            segment_distance(pixel, d * frame_size, a * frame_size))
    );
    return inside * (1.0 - smoothstep(0.65, 1.65, distance));
}

vec2 band_local(
    vec2 p,
    vec2 left_top,
    vec2 right_top,
    vec2 right_bottom,
    vec2 left_bottom,
    vec2 frame_size
) {
    vec2 origin = left_top * frame_size;
    vec2 x_axis = (right_top - left_top) * frame_size;
    x_axis /= max(length(x_axis), 0.001);
    vec2 inward = (0.5 * (right_bottom + left_bottom - right_top - left_top)) * frame_size;
    vec2 y_axis = vec2(-x_axis.y, x_axis.x);
    if (dot(y_axis, inward) < 0.0) {
        y_axis = -y_axis;
    }
    vec2 offset = p * frame_size - origin;
    return vec2(dot(offset, x_axis), dot(offset, y_axis));
}

float glyph_row(float code, float row) {
    float bits = 0.0;
    if (code < 0.5) {
        bits = row < 0.5 ? 17.0 : (row < 1.5 ? 27.0 : (row < 2.5 ? 21.0 : 17.0));
    } else if (code < 1.5) {
        bits = (row < 0.5 || row > 5.5) ? 14.0 : 17.0;
    } else if (code < 2.5) {
        bits = row < 0.5 ? 17.0 : (row < 2.5 ? 25.0 : (row < 3.5 ? 21.0 : (row < 4.5 ? 19.0 : 17.0)));
    } else if (code < 3.5) {
        bits = row < 0.5 ? 30.0 : (row < 2.5 ? 17.0 : (row < 3.5 ? 30.0 : 16.0));
    } else if (code < 4.5) {
        bits = (row < 0.5 || row > 5.5) ? 31.0 : 4.0;
    } else if (code < 5.5) {
        bits = row < 2.0 ? 17.0 : (row < 3.0 ? 10.0 : (row < 4.0 ? 4.0 : (row < 5.0 ? 10.0 : 17.0)));
    } else if (code < 6.5) {
        bits = (row < 0.5 || row > 5.5) ? 31.0 : (row < 3.5 && row > 2.5 ? 30.0 : 16.0);
    } else if (code < 7.5) {
        bits = row > 5.5 ? 31.0 : 16.0;
    } else if (code < 8.5) {
        bits = row < 0.5 ? 14.0 : (row < 3.5 ? 17.0 : (row < 4.5 ? 31.0 : 17.0));
    } else if (code < 9.5) {
        bits = row < 0.5 ? 30.0 : (row < 2.5 ? 17.0 : (row < 3.5 ? 30.0 : (row < 4.5 ? 20.0 : (row < 5.5 ? 18.0 : 17.0))));
    } else if (code < 10.5) {
        bits = row < 0.5 ? 31.0 : 4.0;
    } else if (code < 11.5) {
        bits = (row < 0.5 || row > 5.5) ? 15.0 : 16.0;
    } else {
        bits = (row < 0.5 || row > 5.5 || (row > 2.5 && row < 3.5))
            ? 31.0 : (row < 3.5 ? 16.0 : 1.0);
    }
    return bits;
}

float glyph(float code, vec2 p) {
    if (code < 0.0 || p.x < 0.0 || p.x >= 5.0 || p.y < 0.0 || p.y >= 7.0) {
        return 0.0;
    }
    float column = floor(p.x);
    float bits = glyph_row(code, floor(p.y));
    return mod(floor(bits / exp2(4.0 - column)), 2.0);
}

float label_code(float effect, float index) {
    if (effect < 0.5) {
        if (index < 0.5) return 0.0;
        if (index < 1.5) return 1.0;
        if (index < 2.5) return 2.0;
        if (index < 3.5) return 1.0;
    } else if (effect < 1.5) {
        if (index < 0.5) return 3.0;
        if (index < 1.5) return 4.0;
        if (index < 2.5) return 5.0;
        if (index < 3.5) return 6.0;
        if (index < 4.5) return 7.0;
    } else if (effect < 2.5) {
        if (index < 0.5) return 3.0;
        if (index < 1.5) return 8.0;
        if (index < 2.5) return 9.0;
        if (index < 3.5) return 10.0;
        if (index < 4.5) return 4.0;
        if (index < 5.5) return 11.0;
        if (index < 6.5) return 7.0;
        if (index < 7.5) return 6.0;
    } else {
        if (index < 0.5) return 8.0;
        if (index < 1.5) return 12.0;
        if (index < 2.5) return 11.0;
        if (index < 3.5) return 4.0;
        if (index < 4.5) return 4.0;
    }
    return -1.0;
}

float effect_label(float effect, vec2 local) {
    const float scale = 2.0;
    vec2 text = local - vec2(5.0, 4.0);
    float index = floor(text.x / (6.0 * scale));
    vec2 character = vec2(mod(text.x, 6.0 * scale), text.y) / scale;
    return glyph(label_code(effect, index), character);
}

float label_length(float effect) {
    return effect < 0.5 ? 4.0 : (effect < 1.5 ? 5.0 : (effect < 2.5 ? 8.0 : 5.0));
}

vec2 floating_label_position(float effect, vec2 local) {
    return vec2(local.x + 8.0 + label_length(effect) * 12.0, local.y);
}

float floating_label_region(float effect, vec2 local) {
    float width = 8.0 + label_length(effect) * 12.0;
    return step(-width, local.x) * step(local.x, -2.0) *
        step(1.0, local.y) * step(local.y, 20.0);
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float luminance(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

float ascii_line(vec2 p, vec2 a, vec2 b) {
    return 1.0 - smoothstep(0.045, 0.085, segment_distance(p, a, b));
}

float ascii_mark(float gray, vec2 p) {
    float dot_mark = 1.0 - smoothstep(0.04, 0.085, length(p - vec2(0.5, 0.72)));
    float colon = max(
        1.0 - smoothstep(0.04, 0.08, length(p - vec2(0.5, 0.34))),
        1.0 - smoothstep(0.04, 0.08, length(p - vec2(0.5, 0.68)))
    );
    float plus = max(
        ascii_line(p, vec2(0.25, 0.5), vec2(0.75, 0.5)),
        ascii_line(p, vec2(0.5, 0.25), vec2(0.5, 0.75))
    );
    float star = max(plus, max(
        ascii_line(p, vec2(0.3, 0.3), vec2(0.7, 0.7)),
        ascii_line(p, vec2(0.7, 0.3), vec2(0.3, 0.7))
    ));
    float hash_mark = max(max(
        ascii_line(p, vec2(0.37, 0.2), vec2(0.37, 0.8)),
        ascii_line(p, vec2(0.63, 0.2), vec2(0.63, 0.8))
    ), max(
        ascii_line(p, vec2(0.22, 0.4), vec2(0.78, 0.4)),
        ascii_line(p, vec2(0.22, 0.62), vec2(0.78, 0.62))
    ));
    float at_mark = max(
        1.0 - smoothstep(0.04, 0.08, abs(length((p - 0.5) * vec2(0.9, 1.0)) - 0.3)),
        max(ascii_line(p, vec2(0.52, 0.42), vec2(0.52, 0.65)),
            ascii_line(p, vec2(0.52, 0.62), vec2(0.72, 0.62)))
    );
    return gray < 0.12 ? 0.0 : (gray < 0.27 ? dot_mark :
        (gray < 0.42 ? colon : (gray < 0.57 ? plus :
        (gray < 0.72 ? star : (gray < 0.87 ? hash_mark : at_mark)))));
}

void main() {
    vec2 mask_uv = v_texcoord;
    vec2 uv = vec2(1.0 - mask_uv.x, mask_uv.y);
    vec4 original = texture2D(tex, uv);
    if (active < 0.5) {
        gl_FragColor = original;
        return;
    }

    vec2 l0 = vec2(l0x, l0y);
    vec2 r0 = vec2(r0x, r0y);
    vec2 l1 = vec2(l1x, l1y);
    vec2 r1 = vec2(r1x, r1y);
    vec2 l2 = vec2(l2x, l2y);
    vec2 r2 = vec2(r2x, r2y);
    vec2 l3 = vec2(l3x, l3y);
    vec2 r3 = vec2(r3x, r3y);
    vec2 l4 = vec2(l4x, l4y);
    vec2 r4 = vec2(r4x, r4y);

    float band0 = inside_quad(mask_uv, l0, r0, r1, l1);
    float band1 = inside_quad(mask_uv, l1, r1, r2, l2);
    float band2 = inside_quad(mask_uv, l2, r2, r3, l3);
    float band3 = inside_quad(mask_uv, l3, r3, r4, l4);
    float masked = max(max(band0, band1), max(band2, band3));

    float mode_id = floor(mode + 0.5);
    float effect0 = mod(mode_id, 3.0);
    float effect1 = mod(mode_id + 1.0, 3.0);
    float effect2 = 3.0;
    float effect3 = mod(mode_id + 2.0, 3.0);
    vec2 frame_size = vec2(frame_w, frame_h);
    vec2 local0 = band_local(mask_uv, l0, r0, r1, l1, frame_size);
    vec2 local1 = band_local(mask_uv, l1, r1, r2, l2, frame_size);
    vec2 local2 = band_local(mask_uv, l2, r2, r3, l3, frame_size);
    vec2 local3 = band_local(mask_uv, l3, r3, r4, l4, frame_size);
    float label_region0 = floating_label_region(effect0, local0);
    float label_region1 = floating_label_region(effect1, local1);
    float label_region2 = floating_label_region(effect2, local2);
    float label_region3 = floating_label_region(effect3, local3);
    float label_region = max(max(label_region0, label_region1), max(label_region2, label_region3));
    if (masked < 0.5 && label_region < 0.5) {
        gl_FragColor = original;
        return;
    }

    float label = 0.0;
    float label_shadow = 0.0;
    if (label_region0 > 0.5) {
        vec2 position = floating_label_position(effect0, local0);
        label = max(label, effect_label(effect0, position));
        label_shadow = max(label_shadow, effect_label(effect0, position - vec2(1.0)));
    }
    if (label_region1 > 0.5) {
        vec2 position = floating_label_position(effect1, local1);
        label = max(label, effect_label(effect1, position));
        label_shadow = max(label_shadow, effect_label(effect1, position - vec2(1.0)));
    }
    if (label_region2 > 0.5) {
        vec2 position = floating_label_position(effect2, local2);
        label = max(label, effect_label(effect2, position));
        label_shadow = max(label_shadow, effect_label(effect2, position - vec2(1.0)));
    }
    if (label_region3 > 0.5) {
        vec2 position = floating_label_position(effect3, local3);
        label = max(label, effect_label(effect3, position));
        label_shadow = max(label_shadow, effect_label(effect3, position - vec2(1.0)));
    }

    vec3 final_color = original.rgb;
    if (masked > 0.5) {
        float band_id = band0 > 0.5 ? 0.0 : (band1 > 0.5 ? 1.0 : (band2 > 0.5 ? 2.0 : 3.0));
        float effect_id = band_id < 0.5 ? effect0 :
            (band_id < 1.5 ? effect1 : (band_id < 2.5 ? effect2 : effect3));
        float border;
        if (band_id < 0.5) {
            border = quad_border(mask_uv, l0, r0, r1, l1, frame_size, band0);
        } else if (band_id < 1.5) {
            border = quad_border(mask_uv, l1, r1, r2, l2, frame_size, band1);
        } else if (band_id < 2.5) {
            border = quad_border(mask_uv, l2, r2, r3, l3, frame_size, band2);
        } else {
            border = quad_border(mask_uv, l3, r3, r4, l4, frame_size, band3);
        }

        float displacement = sin(mask_uv.y * frame_h * 0.075 + phase * 3.2) * 0.004;
        vec2 fx_uv = clamp(uv + vec2(displacement, 0.0), vec2(0.0), vec2(1.0));
        vec3 source = texture2D(tex, fx_uv).rgb;
        float gray = luminance(source);
        vec3 mono = vec3(smoothstep(0.28, 0.72, gray));
        vec2 pixel_grid = vec2(54.0, 30.0);
        vec2 pixel_uv = (floor(fx_uv * pixel_grid) + 0.5) / pixel_grid;
        vec3 pixelated = texture2D(tex, pixel_uv).rgb;
        vec2 texel = 1.0 / frame_size;
        float gx = abs(gray - luminance(texture2D(tex, fx_uv + vec2(texel.x, 0.0)).rgb));
        float gy = abs(gray - luminance(texture2D(tex, fx_uv + vec2(0.0, texel.y)).rgb));
        float noise = hash21(floor(fx_uv * frame_size * 0.55) + floor(phase * 12.0));
        float point = step(noise, clamp(gray * 0.42 + (gx + gy) * 4.5, 0.0, 0.9));
        vec3 particles = vec3(point * max(gray, 0.72));
        vec2 ascii_cell = vec2(9.0, 12.0);
        vec2 ascii_pixel = floor(mask_uv * frame_size / ascii_cell) * ascii_cell;
        vec2 ascii_sample = (ascii_pixel + ascii_cell * 0.5) / frame_size;
        ascii_sample.x = 1.0 - ascii_sample.x;
        float ascii_gray = luminance(texture2D(tex, clamp(ascii_sample, vec2(0.0), vec2(1.0))).rgb);
        vec3 ascii = vec3(ascii_mark(ascii_gray, mod(mask_uv * frame_size, ascii_cell) / ascii_cell));
        vec3 effected = effect_id < 0.5 ? mono : (effect_id < 1.5 ? pixelated :
            (effect_id < 2.5 ? particles : ascii));
        float scanline = 0.76 + 0.24 * step(0.5, fract(mask_uv.y * frame_h * 0.25));
        final_color = mix(effected * scanline, vec3(0.94, 0.98, 1.0), border * 0.9);
    }
    final_color = mix(final_color, vec3(0.0), label_shadow * 0.72);
    final_color = mix(final_color, vec3(0.94, 0.98, 1.0), label);
    gl_FragColor = vec4(final_color, 1.0);
}
"""


def _set_float(structure, name, value):
    item = GObject.Value()
    item.init(GObject.TYPE_FLOAT)
    item.set_float(float(value))
    structure.set_value(name, item)


CAMERA_MODES = (
    (1920, 1080, 60),
    (1920, 1080, 30),
    (1280, 720, 60),
    (1280, 720, 30),
    (960, 540, 60),
    (960, 540, 30),
    (640, 480, 60),
    (640, 480, 30),
    (640, 360, 60),
    (640, 360, 30),
    (1920, 1080, 25),
    (1280, 720, 25),
    (640, 480, 25),
    (1920, 1080, 24),
    (1280, 720, 24),
    (640, 480, 24),
    (640, 480, 15),
    (320, 240, 30),
    (320, 240, 15),
)


def _select_camera_mode(caps):
    for width, height, fps in CAMERA_MODES:
        for media_type in ("image/jpeg", "video/x-raw"):
            wanted = Gst.Caps.from_string(
                f"{media_type},width={width},height={height},framerate={fps}/1"
            )
            if caps.can_intersect(wanted):
                return width, height, fps, media_type
    raise RuntimeError("camera has no supported capture mode")


def _camera_backend(camera_index, platform=None):
    platform = platform or sys.platform
    if platform == "win32":
        return "mfvideosrc", "device-index", camera_index
    if platform.startswith("linux"):
        return "v4l2src", "device", f"/dev/video{camera_index}"
    raise RuntimeError(f"camera capture is not supported on {platform}")


def _camera_source(camera_index):
    factory, property_name, value = _camera_backend(camera_index)
    value = f'"{value}"' if isinstance(value, str) else value
    return f"{factory} {property_name}={value} do-timestamp=true"


def _detect_camera_mode_local(camera_index):
    factory, property_name, value = _camera_backend(camera_index)
    source = Gst.ElementFactory.make(factory)
    if source is None:
        raise RuntimeError(f"GStreamer camera plugin {factory!r} is not installed")
    source.set_property(property_name, value)
    try:
        if source.set_state(Gst.State.READY) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"could not open camera index {camera_index}")
        state, _current, _pending = source.get_state(3 * Gst.SECOND)
        if state == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"could not open camera index {camera_index}")
        caps = source.get_static_pad("src").query_caps(None)
        try:
            width, height, fps, media_type = _select_camera_mode(caps)
        except RuntimeError as error:
            raise RuntimeError(
                f"camera index {camera_index} is not a usable capture stream"
            ) from error
        name = (
            source.get_property("device-name")
            if source.find_property("device-name")
            else f"camera {camera_index}"
        )
        return width, height, fps, media_type, name
    finally:
        source.set_state(Gst.State.NULL)


def detect_camera_mode(camera_index):
    result = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--probe-camera",
            str(camera_index),
        ],
        capture_output=True,
        text=True,
        timeout=8,
    )
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(message or f"could not inspect camera index {camera_index}")
    try:
        return tuple(json.loads(result.stdout.strip().splitlines()[-1]))
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid camera probe response: {result.stdout!r}") from error


class GpuRenderer:
    def __init__(
        self,
        width=None,
        height=None,
        fps=None,
        camera_index=None,
        detect_width=640,
        detect_height=None,
        window_width=960,
        headless=False,
    ):
        self.camera_index = camera_index
        self.camera_name = None
        self.media_type = None
        if camera_index is not None:
            if any(value is not None for value in (width, height, fps)):
                raise ValueError("camera size and FPS are auto-detected; set only CAM_INDEX")
            width, height, fps, self.media_type, self.camera_name = detect_camera_mode(
                camera_index
            )
        elif any(value is None for value in (width, height, fps)):
            raise ValueError("test sources require width, height, and fps")

        self.width = width
        self.height = height
        self.fps = fps
        self.detect_width = detect_width
        self.detect_height = detect_height or max(
            2, round(height * detect_width / width / 2) * 2
        )
        self.window_width = min(window_width, width)
        self.window_height = max(
            2, round(height * self.window_width / width / 2) * 2
        )
        self.headless = headless
        self.keys = deque()

        if camera_index is not None:
            source = (
                f"{_camera_source(camera_index)} ! "
                f'{self.media_type},width={width},height={height},framerate={fps}/1'
            )
            if self.media_type == "image/jpeg":
                decoder = (
                    "qsvjpegdec"
                    if Gst.ElementFactory.find("qsvjpegdec")
                    else "jpegdec"
                )
                source += f" ! {decoder}"
        else:
            source = (
                f"videotestsrc is-live=true pattern=gradient ! "
                f"video/x-raw,width={width},height={height},framerate={fps}/1"
            )
        sink = (
            "gldownload ! video/x-raw,format=RGBA ! "
            "appsink name=out sync=false max-buffers=1 drop=true"
            if headless
            else (
                "glcolorscale ! "
                "video/x-raw(memory:GLMemory),format=RGBA,"
                f"width={self.window_width},height={self.window_height} ! "
                "glimagesink name=out sync=false force-aspect-ratio=true"
            )
        )
        description = (
            source
            + " ! tee name=split "
            + "split. ! queue max-size-buffers=1 leaky=downstream "
            + "! glupload ! glcolorconvert "
            + "! video/x-raw(memory:GLMemory),format=RGBA ! glshader name=shader ! "
            + sink
            + " split. ! queue max-size-buffers=1 leaky=downstream "
            + "! videoscale ! videoconvert "
            + f"! video/x-raw,format=RGB,width={self.detect_width},height={self.detect_height} "
            + "! videoflip video-direction=horiz "
            + "! appsink name=detect sync=false max-buffers=1 drop=true"
        )
        self.pipeline = Gst.parse_launch(description)
        self.shader = self.pipeline.get_by_name("shader")
        self.output = self.pipeline.get_by_name("out")
        self.detection = self.pipeline.get_by_name("detect")
        self.bus = self.pipeline.get_bus()
        self.shader.set_property("vertex", VERTEX_SHADER)
        self.shader.set_property("fragment", FRAGMENT_SHADER)
        self.shader.get_static_pad("sink").add_probe(
            Gst.PadProbeType.EVENT_UPSTREAM, self._on_navigation
        )
        self.update(None)

    def camera_info(self):
        if self.camera_index is None:
            return "synthetic camera source"
        codec = "MJPEG" if self.media_type == "image/jpeg" else "raw"
        return (
            f"Camera: {self.camera_name} (index {self.camera_index}) - "
            f"{self.width}x{self.height} @ {self.fps} FPS, {codec}"
        )

    def _on_navigation(self, _pad, info):
        event = info.get_event()
        if GstVideo.navigation_event_get_type(event) == GstVideo.NavigationEventType.KEY_PRESS:
            ok, key = GstVideo.navigation_event_parse_key_event(event)
            if ok:
                self.keys.append(key.lower())
        return Gst.PadProbeReturn.OK

    def start(self):
        last_error = None
        for attempt in range(3):
            result = self.pipeline.set_state(Gst.State.PLAYING)
            message = self.bus.timed_pop_filtered(
                2 * Gst.SECOND,
                Gst.MessageType.ASYNC_DONE | Gst.MessageType.ERROR,
            )
            if result != Gst.StateChangeReturn.FAILURE and (
                message is None or message.type != Gst.MessageType.ERROR
            ):
                return

            if message is not None and message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                last_error = f"{error.message}\n{debug or ''}"
                is_busy = "busy" in error.message.lower()
            else:
                last_error = "pipeline state change failed"
                is_busy = False

            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline.get_state(Gst.SECOND)
            if not is_busy or attempt == 2:
                break
            time.sleep(0.2 * (attempt + 1))

        raise RuntimeError(f"GPU pipeline failed: {last_error}")

    def pop_key(self):
        return self.keys.popleft() if self.keys else None

    def update(self, quads, mode=0, phase=0.0):
        values = {
            "active": 0.0,
            "mode": mode,
            "phase": phase,
            "frame_w": self.width,
            "frame_h": self.height,
        }
        for index in range(5):
            values.update({f"l{index}x": 0.0, f"l{index}y": 0.0})
            values.update({f"r{index}x": 0.0, f"r{index}y": 0.0})

        if quads is not None:
            values["active"] = 1.0
            boundaries = [(quads[0][0], quads[0][1])]
            boundaries.extend((quad[3], quad[2]) for quad in quads)
            for index, (left, right) in enumerate(boundaries):
                values[f"l{index}x"] = left[0] / self.width
                values[f"l{index}y"] = left[1] / self.height
                values[f"r{index}x"] = right[0] / self.width
                values[f"r{index}y"] = right[1] / self.height

        uniforms = Gst.Structure.new_empty("uniforms")
        for name, value in values.items():
            _set_float(uniforms, name, value)
        self.shader.set_property("uniforms", uniforms)

    def _pull(self, sink, shape, timeout):
        sample = sink.emit("try-pull-sample", timeout)
        if sample is None:
            self.check()
            return None
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("could not map video frame")
        try:
            return np.frombuffer(mapped.data, np.uint8).reshape(shape).copy()
        finally:
            buffer.unmap(mapped)

    def pull_detection(self, timeout=Gst.SECOND):
        return self._pull(
            self.detection,
            (self.detect_height, self.detect_width, 3),
            timeout,
        )

    def pull_output(self, timeout=5 * Gst.SECOND):
        if not self.headless:
            raise RuntimeError("pull_output is only available in headless mode")
        output = self._pull(self.output, (self.height, self.width, 4), timeout)
        if output is None:
            raise RuntimeError("GPU pipeline produced no frame")
        return output

    def check(self):
        message = self.bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if message is None:
            return
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            raise RuntimeError(f"GPU pipeline error: {error.message}\n{debug or ''}")
        raise RuntimeError("GPU pipeline stopped")

    def context_info(self):
        context = self.shader.get_property("context")
        if context is None:
            return "OpenGL context unavailable"
        major, minor = GstGL.GLContext.get_gl_version(context)
        platform = GstGL.GLContext.get_gl_platform(context)
        return f"{GstGL.gl_platform_to_string(platform)} OpenGL {major}.{minor}"

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)


def self_check():
    assert _camera_backend(2, "linux") == ("v4l2src", "device", "/dev/video2")
    assert _camera_backend(2, "win32") == ("mfvideosrc", "device-index", 2)
    synthetic_caps = Gst.Caps.from_string(
        "image/jpeg,width=1280,height=720,framerate=60/1;"
        "image/jpeg,width=1920,height=1080,framerate=30/1"
    )
    assert _select_camera_mode(synthetic_caps) == (1920, 1080, 30, "image/jpeg")

    width, height = 320, 180
    boundaries = (
        ((55, 20), (265, 25)),
        ((60, 55), (260, 60)),
        ((65, 90), (255, 95)),
        ((68, 125), (252, 130)),
        ((70, 160), (250, 165)),
    )
    quads = [
        np.float32([left_a, right_a, right_b, left_b])
        for (left_a, right_a), (left_b, right_b) in zip(boundaries, boundaries[1:])
    ]
    renderer = GpuRenderer(width, height, fps=30, headless=True)
    renderer.start()
    try:
        original = renderer.pull_output()
        renderer.update(quads, mode=0, phase=0.5)
        outputs = [renderer.pull_output(), renderer.pull_output()]
        difference = max(
            np.abs(output.astype(np.int16) - original.astype(np.int16)).mean()
            for output in outputs
        )
        print(f"gpu effect difference: {difference:.2f}")
        assert difference > 8
        print(f"gpu self-check ok: {renderer.context_info()}")
    finally:
        renderer.stop()


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--probe-camera":
    try:
        print(json.dumps(_detect_camera_mode_local(int(sys.argv[2]))))
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
