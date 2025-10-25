#pragma comment(linker, "/SUBSYSTEM:WINDOWS")

#define WIN32_LEAN_AND_MEAN

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <windows.h>

#include "draw.h"
#include "nn.h"

#define DEBUG

#define ID_BUTTON_CLEAR     1001
#define ID_BUTTON_PREDICT   1002
#define ID_BUTTON_QUIT      1003

#define ID_BUTTON_SAVE      1004
#define ID_BUTTON_LOAD      1005
#define ID_LOAD_INDEX       1006

#define ID_STATIC_RESULT    1007

NN nn;
ConvLayer conv;
uint8_t nn_initialized = 0;

uint8_t quit = 0;

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    static HDC hMemDC = NULL;
    static HBITMAP hBitmap   = NULL;
    static HGDIOBJ oldBitmap = NULL;

    static uint8_t* data = NULL;
    static uint8_t drawing = 0;

    static HWND hwndButtonClear   = NULL;
    static HWND hwndButtonPredict = NULL;
    static HWND hwndButtonQuit    = NULL;
    static HWND hwndButtonSave    = NULL;
    static HWND hwndButtonLoad    = NULL;
    static HWND hwndLoadIndex     = NULL;
    static HWND hwndStaticResult  = NULL;

	static HFONT hFontButton = NULL;
	static HFONT hFontIndex  = NULL;
	static HFONT hFontResult = NULL;

    switch (msg)
    {
        case WM_CREATE:
        {
            HDC hdc = GetDC(hwnd);
            hMemDC = CreateCompatibleDC(hdc);
            hBitmap = CreateCompatibleBitmap(hdc, CANVAS_SIZE, CANVAS_SIZE);
            oldBitmap = SelectObject(hMemDC, hBitmap);

            data = (uint8_t*)malloc(sizeof(uint8_t) * CELL_LEN * CELL_LEN);

            ClearData(data);

            DrawInCanvas(hMemDC, data);
            ReleaseDC(hwnd, hdc);

            DWORD btnStyle = WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON;

            int32_t btnWidth = (int32_t)(CANVAS_SIZE / 4);
            int32_t btnHeight = (int32_t)(CANVAS_SIZE / 8);

            int32_t btnY = (int32_t)(CANVAS_Y + CANVAS_SIZE + btnHeight / 4);

            hwndButtonClear = CreateWindow(
                "BUTTON", "\nClear", btnStyle, 
                (int)(CANVAS_X + CANVAS_SIZE * 0.0625f), btnY,
                btnWidth, btnHeight, 
                hwnd, (HMENU)ID_BUTTON_CLEAR, NULL, NULL
            );

            hwndButtonPredict = CreateWindow(
                "BUTTON", "\nPredict", btnStyle,
                (int)(CANVAS_X + CANVAS_SIZE * 0.375f), btnY,
                btnWidth, btnHeight,
                hwnd, (HMENU)ID_BUTTON_PREDICT, NULL, NULL
            );

            hwndButtonQuit = CreateWindow(
                "BUTTON", "\nQuit", btnStyle, 
                (int)(CANVAS_X + CANVAS_SIZE * 0.6875f), btnY,
                btnWidth, btnHeight, 
                hwnd, (HMENU)ID_BUTTON_QUIT, NULL, NULL
            );

            hwndButtonSave = CreateWindow(
                "BUTTON", "\nSave", btnStyle,
                (int)(CANVAS_X + CANVAS_SIZE * 0.0625f), (int)(btnY + btnHeight * 1.25),
                btnWidth, btnHeight,
                hwnd, (HMENU)ID_BUTTON_SAVE, NULL, NULL
            );

            hwndButtonLoad = CreateWindow(
                "BUTTON", "\nLoad", btnStyle,
                (int)(CANVAS_X + CANVAS_SIZE * 0.375f), (int)(btnY + btnHeight * 1.25),
                btnWidth, btnHeight,
                hwnd, (HMENU)ID_BUTTON_LOAD, NULL, NULL
            );

            hwndLoadIndex = CreateWindow(
                "EDIT", "0", 
                WS_VISIBLE | WS_CHILD | WS_BORDER | ES_NUMBER | ES_CENTER,
                (int)(CANVAS_X + CANVAS_SIZE * 0.6875f), (int)(btnY + btnHeight * 1.25),
                btnWidth, btnHeight,
                hwnd, (HMENU)ID_LOAD_INDEX, NULL, NULL
            );

            hwndStaticResult = CreateWindow(
                "STATIC", "",
                WS_VISIBLE | WS_CHILD | WS_BORDER | ES_CENTER,
                RESULT_X, RESULT_Y, 
                RESULT_SIZE, RESULT_SIZE, 
                hwnd, (HMENU)ID_STATIC_RESULT, NULL, NULL
            );

            if (!nn_initialized)
                ShowWindow(hwndButtonPredict, SW_HIDE);

#ifndef DEBUG
            ShowWindow(hwndButtonSave, SW_HIDE);
            ShowWindow(hwndButtonLoad, SW_HIDE);
            ShowWindow(hwndLoadIndex, SW_HIDE);
#endif

            HFONT hFontButton = CreateFont(
                (int)(btnHeight / 3),
                (int)(btnHeight / 6),
                0, 0,
                FW_BOLD,
                FALSE, FALSE, FALSE,
                ANSI_CHARSET,
                OUT_DEFAULT_PRECIS,
                CLIP_DEFAULT_PRECIS,
                DEFAULT_QUALITY,
                DEFAULT_PITCH | FF_DONTCARE,
                "Arial"
            );

            HFONT hFontIndex = CreateFont(
                btnHeight,
                (int)(btnHeight / 3),
                0, 0,
                FW_MEDIUM,
                FALSE, FALSE, FALSE,
                ANSI_CHARSET,
                OUT_DEFAULT_PRECIS,
                CLIP_DEFAULT_PRECIS,
                DEFAULT_QUALITY,
                DEFAULT_PITCH | FF_DONTCARE,
                "Arial"
			);

#ifdef DEBUG
            hFontResult = CreateFont(
                (int)(RESULT_SIZE / 12),
                (int)(RESULT_SIZE / 24),
                0, 0,
                FW_BLACK,
                FALSE, FALSE, FALSE,
                ANSI_CHARSET,
                OUT_DEFAULT_PRECIS,
                CLIP_DEFAULT_PRECIS,
                DEFAULT_QUALITY,
                DEFAULT_PITCH | FF_DONTCARE,
                "Arial"
            );
#else
            hFontResult = CreateFont(
                (int)(RESULT_SIZE / 2),
                (int)(RESULT_SIZE / 6),
                0, 0,
                FW_BLACK,
                FALSE, FALSE, FALSE,
                ANSI_CHARSET,
                OUT_DEFAULT_PRECIS,
                CLIP_DEFAULT_PRECIS,
                DEFAULT_QUALITY,
                DEFAULT_PITCH | FF_DONTCARE,
                "Arial"
            );
#endif

            SendMessage(hwndButtonClear, WM_SETFONT, (WPARAM)hFontButton, TRUE);
            SendMessage(hwndButtonPredict, WM_SETFONT, (WPARAM)hFontButton, TRUE);
            SendMessage(hwndButtonQuit, WM_SETFONT, (WPARAM)hFontButton, TRUE);
            SendMessage(hwndButtonSave, WM_SETFONT, (WPARAM)hFontButton, TRUE);
            SendMessage(hwndButtonLoad, WM_SETFONT, (WPARAM)hFontButton, TRUE);

            SendMessage(hwndLoadIndex, WM_SETFONT, (WPARAM)hFontIndex, TRUE);

            SendMessage(hwndStaticResult, WM_SETFONT, (WPARAM)hFontResult, TRUE);

            return 0;
        }

        case WM_SIZE:
        {
            RECT rect;
            GetClientRect(hwnd, &rect);

            SetScreenConstants(
                rect.right - rect.left,
                rect.bottom - rect.top,
#ifdef DEBUG
                1
#else
                0
#endif
            );

            if (hMemDC && hBitmap)
            {
                SelectObject(hMemDC, oldBitmap);
                DeleteObject(hBitmap);
                DeleteDC(hMemDC);

                hMemDC = NULL;
                hBitmap = NULL;
                oldBitmap = NULL;
            }

            HDC hdc = GetDC(hwnd);
            hMemDC = CreateCompatibleDC(hdc);
            hBitmap = CreateCompatibleBitmap(hdc, CANVAS_SIZE, CANVAS_SIZE);
            oldBitmap = SelectObject(hMemDC, hBitmap);

            ReleaseDC(hwnd, hdc);

            DrawInCanvas(hMemDC, data);

            int btnWidth = (int)(CANVAS_SIZE / 4);
            int btnHeight = (int)(CANVAS_SIZE / 8);

            int btnY = (int)(CANVAS_Y + CANVAS_SIZE + btnHeight / 4);

            if (hFontButton)
            {
                DeleteObject(hFontButton);

                hFontButton = NULL;
            }

            if (hFontIndex)
            {
                DeleteObject(hFontIndex);

                hFontIndex = NULL;
            }

            if (hFontResult)
            {
                DeleteObject(hFontResult);

                hFontResult = NULL;
            }

            HFONT hFontButton = CreateFont(
                (int)(btnHeight / 3),
                (int)(btnHeight / 6),
                0, 0,
                FW_BOLD,
                FALSE, FALSE, FALSE,
                ANSI_CHARSET,
                OUT_DEFAULT_PRECIS,
                CLIP_DEFAULT_PRECIS,
                DEFAULT_QUALITY,
                DEFAULT_PITCH | FF_DONTCARE,
                "Arial"
            );

            HFONT hFontIndex = CreateFont(
                btnHeight,
                (int)(btnHeight / 3),
                0, 0,
                FW_MEDIUM,
                FALSE, FALSE, FALSE,
                ANSI_CHARSET,
                OUT_DEFAULT_PRECIS,
                CLIP_DEFAULT_PRECIS,
                DEFAULT_QUALITY,
                DEFAULT_PITCH | FF_DONTCARE,
				"Arial"
			);

#ifdef DEBUG
            hFontResult = CreateFont(
                (int)(RESULT_SIZE / 12),
                (int)(RESULT_SIZE / 24),
                0, 0,
                FW_BLACK,
                FALSE, FALSE, FALSE,
                ANSI_CHARSET,
                OUT_DEFAULT_PRECIS,
                CLIP_DEFAULT_PRECIS,
                DEFAULT_QUALITY,
                DEFAULT_PITCH | FF_DONTCARE,
                "Arial"
            );
#else
            hFontResult = CreateFont(
                (int)(RESULT_SIZE / 2),
                (int)(RESULT_SIZE / 6),
                0, 0,
                FW_BLACK,
                FALSE, FALSE, FALSE,
                ANSI_CHARSET,
                OUT_DEFAULT_PRECIS,
                CLIP_DEFAULT_PRECIS,
                DEFAULT_QUALITY,
                DEFAULT_PITCH | FF_DONTCARE,
                "Arial"
            );
#endif

            if (hwndButtonClear && hwndButtonQuit && hwndButtonPredict && 
                hwndButtonSave && hwndButtonLoad && hwndLoadIndex && 
                hwndStaticResult)
            {
                MoveWindow(
                    hwndButtonClear,
                    (int)(CANVAS_X + CANVAS_SIZE * 0.0625f), btnY,
                    btnWidth, btnHeight,
                    TRUE
                );

                MoveWindow(
                    hwndButtonPredict,
                    (int)(CANVAS_X + CANVAS_SIZE * 0.375f), btnY,
                    btnWidth, btnHeight,
                    TRUE
                );

                MoveWindow(
                    hwndButtonQuit,
                    (int)(CANVAS_X + CANVAS_SIZE * 0.6875f), btnY,
                    btnWidth, btnHeight,
                    TRUE
                );

                MoveWindow(
                    hwndButtonSave,
                    (int)(CANVAS_X + CANVAS_SIZE * 0.0625f), (int)(btnY + btnHeight * 1.25),
                    btnWidth, btnHeight,
                    TRUE
                );

                MoveWindow(
                    hwndButtonLoad,
                    (int)(CANVAS_X + CANVAS_SIZE * 0.375f), (int)(btnY + btnHeight * 1.25),
                    btnWidth, btnHeight,
                    TRUE
                );

                MoveWindow(
                    hwndLoadIndex,
                    (int)(CANVAS_X + CANVAS_SIZE * 0.6875f), (int)(btnY + btnHeight * 1.25),
                    btnWidth, btnHeight,
                    TRUE
                );

                MoveWindow(
                    hwndStaticResult,
                    RESULT_X, RESULT_Y, 
                    RESULT_SIZE, RESULT_SIZE,
                    TRUE
                );

                SendMessage(hwndButtonClear, WM_SETFONT, (WPARAM)hFontButton, TRUE);
                SendMessage(hwndButtonPredict, WM_SETFONT, (WPARAM)hFontButton, TRUE);
                SendMessage(hwndButtonQuit, WM_SETFONT, (WPARAM)hFontButton, TRUE);
                SendMessage(hwndButtonSave, WM_SETFONT, (WPARAM)hFontButton, TRUE);
                SendMessage(hwndButtonLoad, WM_SETFONT, (WPARAM)hFontButton, TRUE);
                
                SendMessage(hwndLoadIndex, WM_SETFONT, (WPARAM)hFontIndex, TRUE);
                
                SendMessage(hwndStaticResult, WM_SETFONT, (WPARAM)hFontResult, TRUE);
            }

            InvalidateRect(hwnd, NULL, TRUE);
            UpdateWindow(hwnd);

            return 0;
        }

        case WM_LBUTTONDOWN:
        {
            drawing = 1;

            return 0;
        }

        case WM_MOUSEMOVE:
        {
            if (drawing && hMemDC != NULL)
            {
                int x = LOWORD(lParam);
                int y = HIWORD(lParam);

                DrawCircleInCanvas(data, x, y);
                DrawInCanvas(hMemDC, data);

                RECT crect = { CANVAS_X, CANVAS_Y, CANVAS_X + CANVAS_SIZE, CANVAS_Y + CANVAS_SIZE };
                InvalidateRect(hwnd, &crect, FALSE);
            }

            return 0;
        }

        case WM_LBUTTONUP:
        {
            drawing = 0;

            return 0;
        }

        case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

            if (hMemDC != NULL && data != NULL && CANVAS_SIZE > 0)
                BitBlt(
                    hdc,
                    CANVAS_X, CANVAS_Y,
                    CANVAS_SIZE, CANVAS_SIZE,
                    hMemDC, 0, 0, SRCCOPY
                );

            EndPaint(hwnd, &ps);

            return 0;
        }

        case WM_CTLCOLORSTATIC:
        {
            HDC hdcStatic = (HDC)wParam;
            HWND hwndCtl = (HWND)lParam;

            if (hwndCtl == hwndStaticResult)
            {
                SetTextColor(hdcStatic, RGB(255, 255, 255));
                SetBkColor(hdcStatic, RGB(0, 0, 0));

                static HBRUSH hBrush = NULL;

                if (!hBrush)
                    hBrush = CreateSolidBrush(RGB(0, 0, 0));

                return (INT_PTR)hBrush;
            }

            break;
        }

        case WM_COMMAND:
        {
            switch (LOWORD(wParam))
            {
                case ID_BUTTON_CLEAR:
                {
                    if (hMemDC && data)
                    {
                        ClearData(data);
                        DrawInCanvas(hMemDC, data);

                        InvalidateRect(hwnd, NULL, TRUE);
                    }

                    break;
                }

                case ID_BUTTON_QUIT:
                {
                    quit = 1;

                    PostQuitMessage(0);
                    
                    break;
                }

                case ID_BUTTON_PREDICT:
                {
                    if (data != NULL && nn_initialized)
                    {
                        size_t img_size = CELL_LEN;
                        Mat input_image = Mat_alloc(img_size, img_size);

                        // Normalize canvas data (0~1)
                        for (size_t y = 0; y < img_size; y++)
                            for (size_t x = 0; x < img_size; x++)
                                MAT_AT(input_image, y, x) = data[y * img_size + x] / 255.0f;
                        
                        size_t conv_out_h = img_size - conv.kernel_size + 1;
                        size_t conv_out_w = img_size - conv.kernel_size + 1;
                        
                        size_t pooled_h = conv_out_h / 2;
                        size_t pooled_w = conv_out_w / 2;

                        // Allocate conv_out and pooled
                        Mat* conv_out = malloc(sizeof(Mat) * conv.out_channels);
                        Mat* pooled = malloc(sizeof(Mat) * conv.out_channels);

                        if (!conv_out || !pooled)
                        {
                            free(conv_out);
                            free(pooled);

                            Mat_free(input_image);

                            ShowMessage("Memory allocation failed for conv_out or pooled.", TYPE_ERROR);
                            
                            break;
                        }

                        int alloc_failed = 0;

                        for (size_t c = 0; c < conv.out_channels; c++)
                        {
                            conv_out[c] = Mat_alloc(conv_out_h, conv_out_w);
                            pooled[c] = Mat_alloc(pooled_h, pooled_w);

                            if (!conv_out[c].es || !pooled[c].es)
                            {
                                alloc_failed = 1;

                                break;
                            }
                        }

                        if (alloc_failed)
                        {
                            for (size_t c = 0; c < conv.out_channels; c++)
                            {
                                Mat_free(conv_out[c]);
                                Mat_free(pooled[c]);
                            }

                            free(conv_out);
                            free(pooled);

                            Mat_free(input_image);

                            ShowMessage("Mat allocation failed", TYPE_ERROR);
                            
                            break;
                        }

                        // CNN forward
                        Mat flat;

                        CNN_forward(nn, conv, input_image, conv_out, pooled, &flat);

                        float max_prob = 0.0f;
                        int max_digit = 0;

                        // Find the digit with the highest probability
                        for (int i = 0; i < 10; i++)
                        {
                            float prob = MAT_AT(NN_OUTPUT(nn), 0, i);

                            if (prob > max_prob)
                            {
                                max_prob = prob;
                                max_digit = i;
                            }
                        }

                        // Prepare result text
                        char result_text[256] = { 0 };

                        char buf[16];

#ifndef DEBUG
                        if (max_prob < 0.75f)
                            snprintf(buf, sizeof(buf), "%d?\n(%d%%)", max_digit, (int)(max_prob * 100));
                        else
                            snprintf(buf, sizeof(buf), "%d\n(%d%%)", max_digit, (int)(max_prob * 100));

						strcat_s(result_text, sizeof(result_text), buf);
#else
                        if (max_prob < 0.75f)
                            snprintf(buf, sizeof(buf), "%d? (%d%%)\n", max_digit, (int)(max_prob * 100));
                        else
                            snprintf(buf, sizeof(buf), "%d (%d%%)\n", max_digit, (int)(max_prob * 100));

                        strcat_s(result_text, sizeof(result_text), buf);

                        for (int i = 0; i < 10; i++)
                        {                            
                            snprintf(buf, sizeof(buf), "\nDigit %d: %.3f", i, MAT_AT(NN_OUTPUT(nn), 0, i));
                            
                            strcat_s(result_text, sizeof(result_text), buf);
                        }
#endif

                        SetWindowText(hwndStaticResult, result_text);

                        // Free memory
                        for (size_t c = 0; c < conv.out_channels; c++)
                        {
                            Mat_free(conv_out[c]);
                            Mat_free(pooled[c]);
                        }

                        free(conv_out);
                        free(pooled);

                        Mat_free(flat);
                        Mat_free(input_image);
                    }

                    break;
                }

                case ID_BUTTON_SAVE:
                {
                    if (data != NULL)
                    {
                        FILE* file = NULL;
                        size_t count = 0;

                        fopen_s(&file, "data/number.dat", "r+b");

                        if (!file)
                        {
                            fopen_s(&file, "data/number.dat", "w+b");

                            if (!file)
                            {
                                ShowMessage("Failed to create 'data/number.dat'", TYPE_ERROR);

                                break;
                            }

                            fwrite("NUMDATA", sizeof(char), 7, file);

                            fwrite(&count, sizeof(size_t), 1, file);
                        }
                        else
                        {
                            char header_buf[7];
                            fread(header_buf, sizeof(char), 7, file);

                            if (memcmp(header_buf, "NUMDATA", 7) != 0)
                            {
                                fclose(file);


                                ShowMessage("File header mismatch in 'data/number.dat'", TYPE_ERROR);

                                break;
                            }

                            fread(&count, sizeof(size_t), 1, file);
                        }

                        fseek(file, 0, SEEK_END);
                        fwrite(data, sizeof(uint8_t), (size_t)(CELL_LEN * CELL_LEN), file);

                        count++;
                        fseek(file, 7, SEEK_SET);
                        fwrite(&count, sizeof(size_t), 1, file);

                        fclose(file);

                        ClearData(data);
                        DrawInCanvas(hMemDC, data);

                        InvalidateRect(hwnd, NULL, TRUE);
                    }

                    break;
                }

                case ID_BUTTON_LOAD:
                {
                    if (data != NULL)
                    {
                        char buf[32] = { 0 };
                        int idx = 0;

                        GetWindowText(hwndLoadIndex, buf, sizeof(buf));
                        idx = atoi(buf);

                        FILE* file = NULL;
                        fopen_s(&file, "data/number.dat", "rb");
                        
                        if (!file)
                        {
                            ShowMessage("Failed to open 'data/number.dat'", TYPE_ERROR);
                            
                            break;
                        }

                        char header[7];
                        fread(header, sizeof(char), 7, file);

                        if (memcmp(header, "NUMDATA", 7) != 0)
                        {
                            fclose(file);

                            ShowMessage("File header mismatch", TYPE_ERROR);
                            
                            break;
                        }

                        size_t count = 0;
                        fread(&count, sizeof(size_t), 1, file);

                        if (idx < 0 || (size_t)idx >= count)
                        {
                            fclose(file);

                            ShowMessage("Invalid dataset index", TYPE_ERROR);

                            break;
                        }

                        fseek(file, 7 + sizeof(size_t) + (size_t)(idx * CELL_LEN * CELL_LEN), SEEK_SET);
                        fread(data, sizeof(uint8_t), (size_t)(CELL_LEN * CELL_LEN), file);

                        fclose(file);

                        if (hMemDC)
                            DrawInCanvas(hMemDC, data);

                        InvalidateRect(hwnd, NULL, TRUE);
                    }

                    break;
                }
            }

            return 0;
        }

        case WM_DESTROY:
        {
            if (hMemDC != NULL)
            {
                SelectObject(hMemDC, oldBitmap);
                DeleteObject(hBitmap);
                DeleteDC(hMemDC);

                hMemDC = NULL;
                hBitmap = NULL;
                oldBitmap = NULL;
            }

            if (data != NULL)
            {
                free(data);

                data = NULL;
            }

            if (hFontResult)
            {
                DeleteObject(hFontResult);

                hFontResult = NULL;
            }

            Conv_free(&conv);

            PostQuitMessage(0);

            return 0;
        }
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

int WINAPI WinMain(
    _In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine,
    _In_ int nCmdShow
)
{
    (void)hPrevInstance;
    (void)lpCmdLine;

    FILE* file = fopen("data/model.cnn", "rb");

    if (file)
    {
        CNN_load(&conv, &nn, file);

        fclose(file);

        nn_initialized = 1;
    }
    else
    {
        ShowMessage("Could not open 'data/model.cnn'.", TYPE_ERROR);
        ShowMessage("Prediction will be disabled.", TYPE_INFO);
    }

    WNDCLASS wc = { 0 };
    
    wc.style = 0;
    wc.lpfnWndProc = (WNDPROC) WndProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;

    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    
    wc.hbrBackground = CreateSolidBrush(RGB(32, 32, 32));
    
    wc.lpszMenuName = NULL;
    wc.lpszClassName = "Number_Recognition";

    if (!RegisterClass(&wc))
    {
        if (nn_initialized)
        {
            NN_free(&nn);
            ZeroMemory(&nn, sizeof(nn));
        }

        return EXIT_FAILURE;
    }

    SetScreenConstants(
        1600, 
        900,
#ifdef DEBUG
        1
#else
        0
#endif
    );


    HWND hWnd = CreateWindow(
        "Number_Recognition", "Number Recognition",
        WS_SIZEBOX,
        0, 0, SCREEN_WIDTH, SCREEN_HEIGHT,
        NULL, NULL, hInstance, NULL
    );

    if (!hWnd)
    {
        ShowMessage("Failed to create window", TYPE_ERROR);

        if (nn_initialized)
        {
            NN_free(&nn);
            ZeroMemory(&nn, sizeof(nn));
        }

        return EXIT_FAILURE;
    }

	FreeConsole();

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    MSG msg;
    while (!quit && GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    if (nn_initialized)
    {
        NN_free(&nn);
        ZeroMemory(&nn, sizeof(nn));
    }

    return EXIT_SUCCESS;
}