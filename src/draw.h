#pragma once

#define WIN32_LEAN_AND_MEAN

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>

#define TYPE_ERROR      1
#define TYPE_WARNING    2
#define TYPE_INFO       3

int32_t SCREEN_WIDTH, SCREEN_HEIGHT,
        SCREEN_STANDARD,
        CANVAS_SIZE, CANVAS_X, CANVAS_Y,
        CELL_LEN, CELL_SIZE, 
        RESULT_SIZE, RESULT_X, RESULT_Y;

void SetScreenConstants(int32_t screen_width, int32_t screen_height, uint8_t DEBUG);

void ShowMessage(const char* message, int type);

void ClearData(uint8_t* data);

void DrawInCanvas(HDC hdc, uint8_t* data);
void DrawCircleInCanvas(uint8_t* data, int x, int y);