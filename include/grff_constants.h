#pragma once

// Sizes from GRFF/source/MWtransfer.h
#define LpSize 5
#define InSize 15
#define OutSize 7
#define RpSize 3
#define RpSize1 9
#define GpSize 7

// D2/D3 from GRFF/source/ExtInterface.h
#define D2(s1, i1, i2) ((i1) + (i2) * (s1))
#define D3(s1, s2, i1, i2, i3) ((i1) + ((i2) + (i3) * (s2)) * (s1))
