#include <stdio.h>
#include <string.h>

#include "esp_camera.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

#include "dl_image_define.hpp"
#include "dl_image_pixel_cvt_dispatch.hpp"
#include "coco_detect.hpp"

static const char *TAG = "bird_detect";

// Proven camera pin mapping
#define CAM_XCLK    15
#define CAM_SIOD     4
#define CAM_SIOC     5

#define CAM_D0      11
#define CAM_D1       9
#define CAM_D2       8
#define CAM_D3      10
#define CAM_D4      12
#define CAM_D5      18
#define CAM_D6      17
#define CAM_D7      16

#define CAM_VSYNC    6
#define CAM_HREF     7
#define CAM_PCLK    13

#define CAM_PWDN    -1
#define CAM_RESET   -1

#define CAMERA_WIDTH   320
#define CAMERA_HEIGHT  240

// RGB888 = 3 bytes per pixel
#define RGB888_BUFFER_SIZE \
    (CAMERA_WIDTH * CAMERA_HEIGHT * 3)

static esp_err_t init_camera()
{
    camera_config_t config = {};

    config.pin_pwdn = CAM_PWDN;
    config.pin_reset = CAM_RESET;
    config.pin_xclk = CAM_XCLK;
    config.pin_sccb_sda = CAM_SIOD;
    config.pin_sccb_scl = CAM_SIOC;

    config.pin_d7 = CAM_D7;
    config.pin_d6 = CAM_D6;
    config.pin_d5 = CAM_D5;
    config.pin_d4 = CAM_D4;
    config.pin_d3 = CAM_D3;
    config.pin_d2 = CAM_D2;
    config.pin_d1 = CAM_D1;
    config.pin_d0 = CAM_D0;

    config.pin_vsync = CAM_VSYNC;
    config.pin_href = CAM_HREF;
    config.pin_pclk = CAM_PCLK;

    config.xclk_freq_hz = 20000000;
    config.ledc_timer = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;

    // ESP32-S3 camera path supports RGB565.
    // RGB888 conversion is done in software below.
    config.pixel_format = PIXFORMAT_RGB565;

    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.jpeg_buffer_size = 0;

    esp_err_t err = esp_camera_init(&config);

    if (err != ESP_OK) {
        ESP_LOGE(TAG,
                 "Camera initialization failed: 0x%x",
                 err);
        return err;
    }

    ESP_LOGI(TAG, "Camera initialized successfully");

    return ESP_OK;
}

static void send_debug_jpeg(camera_fb_t *fb)
{
    uint8_t *jpg_buf = nullptr;
    size_t jpg_len = 0;

    if (!frame2jpg(fb, 80, &jpg_buf, &jpg_len)) {
        ESP_LOGE(TAG, "JPEG conversion failed");
        return;
    }

    ESP_LOGI(TAG, "JPEG debug frame: %u bytes", (unsigned)jpg_len);

    printf("JPEG_BEGIN\n");

    // Base64 encode so binary JPEG data doesn't corrupt the serial monitor.
    static const char b64[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    for (size_t i = 0; i < jpg_len; i += 3) {
        uint32_t v = jpg_buf[i] << 16;

        if (i + 1 < jpg_len)
            v |= jpg_buf[i + 1] << 8;

        if (i + 2 < jpg_len)
            v |= jpg_buf[i + 2];

        putchar(b64[(v >> 18) & 0x3F]);
        putchar(b64[(v >> 12) & 0x3F]);
        putchar(i + 1 < jpg_len ? b64[(v >> 6) & 0x3F] : '=');
        putchar(i + 2 < jpg_len ? b64[v & 0x3F] : '=');
    }

    printf("\nJPEG_END\n");
    fflush(stdout);

    free(jpg_buf);
}
extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting bird detector...");

    esp_err_t err = init_camera();

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera initialization failed");
        return;
    }
    
    /*
     * Allocate one RGB888 buffer and reuse it for every frame.
     *
     * 320 × 240 × 3 = 230,400 bytes
     *
     * PSRAM is used because this is a relatively large buffer.
     */
    uint8_t *rgb888_buffer =
        static_cast<uint8_t *>(
            heap_caps_malloc(
                RGB888_BUFFER_SIZE,
                MALLOC_CAP_SPIRAM
            )
        );

    if (rgb888_buffer == nullptr) {
        ESP_LOGE(TAG,
                 "Failed to allocate RGB888 buffer (%u bytes)",
                 (unsigned)RGB888_BUFFER_SIZE);
        return;
    }

    ESP_LOGI(TAG,
             "RGB888 buffer allocated in PSRAM: %u bytes",
             (unsigned)RGB888_BUFFER_SIZE);

    ESP_LOGI(TAG, "Creating ESP-DL bird detector...");

    COCODetect detector(COCODetect::YOLO11N_320_S8_V1, true);

    ESP_LOGI(TAG, "Bird detector created");


    while (true) {

        /*
         * 1. Capture RGB565 frame from OV3660.
         */
        camera_fb_t *fb = esp_camera_fb_get();

        if (fb == nullptr) {
            ESP_LOGE(TAG, "Failed to capture camera frame");
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        ESP_LOGI(
            TAG,
            "Frame: %ux%u format=%d bytes=%u",
            (unsigned)fb->width,
            (unsigned)fb->height,
            (int)fb->format,
            (unsigned)fb->len
        );

        /*
         * Verify the camera returned exactly the format/size
         * expected by the conversion below.
         */
        if (fb->format != PIXFORMAT_RGB565 ||
            fb->width != CAMERA_WIDTH ||
            fb->height != CAMERA_HEIGHT) {

            ESP_LOGE(
                TAG,
                "Unexpected frame: %ux%u format=%d",
                (unsigned)fb->width,
                (unsigned)fb->height,
                (int)fb->format
            );

            esp_camera_fb_return(fb);
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        /*
         * 2. Convert:
         *
         *     RGB565 big-endian
         *             ↓
         *          RGB888
         *
         * The OV3660/esp32-camera framebuffer byte ordering
         * was verified against Espressif's own RGB565 conversion.
         */
        // NOTE: dl::image::cvt_pix converts ONE pixel only, so convert the
        // whole frame here. esp32-camera RGB565 is big-endian:
        //   byte0 = RRRRRGGG, byte1 = GGGBBBBB
        {
            const uint8_t *src = fb->buf;
            uint8_t *dst = rgb888_buffer;
            const size_t n = (size_t)CAMERA_WIDTH * CAMERA_HEIGHT;
            for (size_t p = 0; p < n; p++) {
                uint16_t v = ((uint16_t)src[2 * p] << 8) | src[2 * p + 1];
                uint8_t r = (v >> 11) & 0x1F;
                uint8_t g = (v >> 5) & 0x3F;
                uint8_t b = v & 0x1F;
                dst[3 * p + 0] = (r << 3) | (r >> 2);
                dst[3 * p + 1] = (g << 2) | (g >> 4);
                dst[3 * p + 2] = (b << 3) | (b >> 2);
            }
        }
        static int frame_no = 0;
        frame_no++;
        if (frame_no == 1 || frame_no == 25) {
           send_debug_jpeg(fb);
        }
        /*
         * 3. Create ESP-DL image descriptor around the
         *    converted RGB888 buffer.
         */
        dl::image::img_t img = {
            rgb888_buffer,
            static_cast<uint16_t>(fb->width),
            static_cast<uint16_t>(fb->height),
            dl::image::DL_IMAGE_PIX_TYPE_RGB888
        };

        /*
         * The camera framebuffer is no longer needed after
         * the conversion. Return it before running inference.
         */
        esp_camera_fb_return(fb);

        /*
         * 4. Run bird detection on RGB888 image.
         */
        /* RGB888 diagnostic */
    uint32_t sum_r = 0;
    uint32_t sum_g = 0;
    uint32_t sum_b = 0;
    uint8_t min_v = 255;
    uint8_t max_v = 0;

    for (size_t i = 0; i < RGB888_BUFFER_SIZE; i += 3) {
    uint8_t r = rgb888_buffer[i];
    uint8_t g = rgb888_buffer[i + 1];
    uint8_t b = rgb888_buffer[i + 2];

    sum_r += r;
    sum_g += g;
    sum_b += b;

    if (r < min_v) min_v = r;
    if (g < min_v) min_v = g;
    if (b < min_v) min_v = b;

    if (r > max_v) max_v = r;
    if (g > max_v) max_v = g;
    if (b > max_v) max_v = b;
    }

   size_t pixel_count = CAMERA_WIDTH * CAMERA_HEIGHT;

    ESP_LOGI(
    TAG,
    "RGB888 stats: R=%u G=%u B=%u min=%u max=%u",
    (unsigned)(sum_r / pixel_count),
    (unsigned)(sum_g / pixel_count),
    (unsigned)(sum_b / pixel_count),
    (unsigned)min_v,
    (unsigned)max_v
    );

     ESP_LOGI(TAG, "Running inference...");
        auto &results = detector.run(img);

        ESP_LOGI(
            TAG,
            "Detections: %u",
            (unsigned)results.size()
        );

        /*
         * 5. Report detections.
         */
        // COCO class 14 = "bird". Everything else is ignored.
        static const int BIRD_CLASS = 14;
        const float CONFIRM_SCORE = 0.40f;
        const int CONFIRM_FRAMES = 2;
        static int consecutive = 0;
        float best = 0.0f;
        for (const auto &result : results) {
            ESP_LOGI(TAG, "seen class=%d score=%.2f box=[%d,%d,%d,%d]%s",
                     result.category, result.score,
                     result.box[0], result.box[1], result.box[2], result.box[3],
                     result.category == BIRD_CLASS ? "  <-- bird" : "");
            if (result.category == BIRD_CLASS && result.score > best) best = result.score;
        }
        consecutive = (best >= CONFIRM_SCORE) ? consecutive + 1 : 0;
        if (consecutive >= CONFIRM_FRAMES) {
            ESP_LOGW(TAG, "BIRD DETECTED - score %.2f (%d frames in a row)", best, consecutive);
        }

        /*
         * rgb888_buffer is intentionally NOT freed here.
         *
         * It is reused for the next camera frame.
         */
        vTaskDelay(pdMS_TO_TICKS(50));
    }

    /*
     * This point is never reached because the application
     * continuously processes camera frames.
     *
     * If the loop is later changed to terminate, free the
     * reusable RGB888 buffer here.
     */
    heap_caps_free(rgb888_buffer);
}
