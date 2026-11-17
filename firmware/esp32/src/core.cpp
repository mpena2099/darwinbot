// PlatformIO only compiles what is under src/, so this is the one line that
// puts the portable core in the image: the file itself, never a copy of it.
#include "../core/darwin_control.cpp"
