cmake_minimum_required (VERSION 3.8)
project ("Digit_Recognition")

if(MSVC)
    add_compile_options(/W3 /WX)
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2)
    else()
        add_compile_options(/Od /RTC1)
    endif()
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -Wall -Wextra -Werror)
    else()
        add_compile_options(-O0 -g -Wall -Wextra -Werror)
    endif()
endif()

add_executable(
    train
    src/train.c
    "src/nn.h" "src/nn.c"
)

add_executable(
    predict
    "src/predict.c"
    "src/draw.h" "src/draw.c"
    "src/nn.h" "src/nn.c"
)