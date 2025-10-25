#include "draw.h"

void SetScreenConstants(int32_t screen_width, int32_t screen_height, uint8_t DEBUG)
{
    SCREEN_WIDTH = (int32_t)screen_width;
    SCREEN_HEIGHT = (int32_t)screen_height;

    if (SCREEN_WIDTH > SCREEN_HEIGHT)
    {
        CANVAS_SIZE =
            (SCREEN_WIDTH / 3.0f < SCREEN_HEIGHT / (1 + 0.3125f / (2 - DEBUG))) ?
            (int32_t)(SCREEN_WIDTH / 3) :
            (int32_t)(SCREEN_HEIGHT / (1 + 0.3125f / (2 - DEBUG)));

        CANVAS_X = (int32_t)((SCREEN_WIDTH - CANVAS_SIZE * 2) / 3);
        CANVAS_Y = (int32_t)((SCREEN_HEIGHT - CANVAS_SIZE * (1 + 0.3125f / (2 - DEBUG))) / 2);

		RESULT_SIZE = CANVAS_SIZE;

		RESULT_X = (int32_t)(SCREEN_WIDTH - CANVAS_X - CANVAS_SIZE);
		RESULT_Y = CANVAS_Y;
    }
    else
    {
        CANVAS_SIZE =
            (SCREEN_HEIGHT / (3 + 0.3125f / (2 - DEBUG)) < SCREEN_WIDTH) ?
            (int32_t)(SCREEN_HEIGHT / (3 + 0.3125f / (2 - DEBUG))) :
            SCREEN_WIDTH;

        CANVAS_X = (int32_t)((SCREEN_WIDTH - CANVAS_SIZE) / 2);
        CANVAS_Y = (int32_t)((SCREEN_HEIGHT - CANVAS_SIZE * (2 + 0.3125f / (2 - DEBUG))) / 3);

		RESULT_SIZE = CANVAS_SIZE;

		RESULT_X = CANVAS_X;
		RESULT_Y = (int32_t)(SCREEN_HEIGHT - CANVAS_Y - CANVAS_SIZE);
    }

    CELL_LEN = 28;
    CELL_SIZE = (int32_t)(CANVAS_SIZE / CELL_LEN);
}

void ShowMessage(const char* message, int type)
{
    if (type == TYPE_ERROR)
        MessageBoxA(NULL, message, "Error", MB_ICONERROR | MB_OK);

    else if (type == TYPE_WARNING)
        MessageBoxA(NULL, message, "Warning", MB_ICONWARNING | MB_OK);

    else if (type == TYPE_INFO)
        MessageBox(NULL, message, "Info", MB_ICONINFORMATION | MB_OK);
}

void ClearData(uint8_t* data)
{
    for (uint16_t i = 0; i < CELL_LEN * CELL_LEN; i++)
        data[i] = 0;
}

void DrawInCanvas(HDC hdc, uint8_t* data)
{
    static uint32_t* dib = NULL;
    static int dib_pixels = 0;

    if (CANVAS_SIZE <= 0 || CELL_SIZE <= 0 || data == NULL) return;

    int total_pixels = CANVAS_SIZE * CANVAS_SIZE;

    if (dib_pixels != total_pixels)
    {
        free(dib);

        dib = (uint32_t*)malloc(sizeof(uint32_t) * total_pixels);
        dib_pixels = total_pixels;
    }

    if (dib == NULL) return;

    BITMAPINFO bmi;
    ZeroMemory(&bmi, sizeof(bmi));

    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = CANVAS_SIZE;
    bmi.bmiHeader.biHeight = -CANVAS_SIZE;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    if (data != NULL)
    {
        for (int py = 0; py < CANVAS_SIZE; py++)
        {
            int cell_y = py / CELL_SIZE;
            if (cell_y >= CELL_LEN) cell_y = CELL_LEN - 1;

            for (int px = 0; px < CANVAS_SIZE; px++)
            {
                int cell_x = px / CELL_SIZE;
                if (cell_x >= CELL_LEN) cell_x = CELL_LEN - 1;

                uint8_t v = data[cell_y * CELL_LEN + cell_x];
                uint32_t col = 0xFF000000 | (v << 16) | (v << 8) | v;

                dib[py * CANVAS_SIZE + px] = col;
            }
        }

        SetDIBitsToDevice(
            hdc,
            0, 0,
            CANVAS_SIZE, CANVAS_SIZE,
            0, 0,
            0, CANVAS_SIZE,
            dib,
            &bmi,
            DIB_RGB_COLORS
        );
    }
}

void DrawCircleInCanvas(uint8_t* data, int x, int y)
{
    if (x < CANVAS_X || x >= CANVAS_X + CANVAS_SIZE ||
        y < CANVAS_Y || y >= CANVAS_Y + CANVAS_SIZE)
        return;

    int cell_x = (x - CANVAS_X) / CELL_SIZE;
    int cell_y = (y - CANVAS_Y) / CELL_SIZE;

    const int radius = 2;

    for (int dy = -radius; dy <= radius; dy++)
    {
        for (int dx = -radius; dx <= radius; dx++)
        {
            int cx = cell_x + dx;
            int cy = cell_y + dy;

            if (cx < 0 || cx >= CELL_LEN || cy < 0 || cy >= CELL_LEN)
                continue;

            float dist = sqrtf((float)(dx * dx + dy * dy));
            if (dist > radius)
                continue;

            float intensity = 1.0f - (dist / radius);
            if (intensity < 0.0f) intensity = 0.0f;

            int idx = cy * CELL_LEN + cx;
            int v = data[idx] + (int)(intensity * 64);

            if (v > 255) v = 255;
            data[idx] = (uint8_t)v;
        }
    }
}