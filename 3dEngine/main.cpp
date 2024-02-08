
#include "render.hpp"
#define IDT_TIMER1 1
// Define a window procedure function to handle messages
LONG64 i = 0;
float cameraX = 0.0f; 
float cameraY = 0.0f; 
float cameraZ = 0.0f; 
std::string cameraName = "MainCamera"; 
float fov = 90.0f; //
float aspectRatio = 16.0f / 9.0f; 
float nearPlane = 0.1f; // Near clipping plane
float farPlane = 100.0f; // Far clipping plane
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE:
        // Set up a timer that ticks every 100 milliseconds (1/10th of a second)
        if (SetTimer(hwnd, IDT_TIMER1, 1, NULL) != 0) {
            break;
        }
        return 0;

    case WM_TIMER:
        switch (wParam)
        {
        case IDT_TIMER1:
            // Invalidate the entire window, causing WM_PAINT to be sent
            if (InvalidateRect(hwnd, NULL, TRUE) != 0) {
                break;
            }
            break;
        }
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_PAINT:
    {
        camera cam(cameraX, cameraY, cameraZ, cameraName, fov, aspectRatio, nearPlane, farPlane);
        i++;
        PAINTSTRUCT ps;
        mesh cube = CreateCube(0.4f, 0.5f, 1.0f, 0.5f);
        mesh cube2 = CreateCube(0.0f, 0.0f, 1.0f, 0.5f);


        transform(cube, -0.0f, -0.0f, 2.0f);
        transform(cube2, -0.0f, -0.0f, 2.0f);

        rotate(cube,0, 0, 0);

        point2D test(-0.1, 0);
        RECT rect;
        int width;
        int height;
        HDC hdc = BeginPaint(hwnd, &ps);
        if (GetWindowRect(hwnd, &rect))
        {
            width = rect.right - rect.left;
            height = rect.bottom - rect.top;
        }
        else {
            break;
        }
        HDC hdesktop = GetDC(0);
        HDC memdc = CreateCompatibleDC(hdesktop);
       
        HBITMAP hbitmap = CreateCompatibleBitmap(hdesktop, width, height);
        
        SelectObject(memdc, hbitmap);
        // Fill the bitmap with red color
        COLORREF redColor = RGB(255, 0, 0);
        fixPoint(test, width, height);
        //SetPixel(memdc,  test.x, test.y, redColor);
       DrawMesh(memdc, cube, redColor, width, height, cam);
       DrawMesh(memdc, cube2, redColor, width, height, cam);

        


        BitBlt(hdc, 0, 0, width, height, memdc, 0, 0, SRCCOPY);
        EndPaint(hwnd, &ps);




    }
    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR pCmdLine, int nCmdShow) {
    // Step 1: Register the Window Class
    const wchar_t CLASS_NAME[] = L"Sample Window Class";

    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME; // Correct assignment here

    RegisterClass(&wc);

    // Step 2: Create the Window
    HWND hwnd = CreateWindowEx(
        0,                              // Optional window styles.
        CLASS_NAME,                     // Window class
        NULL,    // Window text
        WS_OVERLAPPEDWINDOW,            // Window style

        // Size and position
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

        NULL,       // Parent window    
        NULL,       // Menu
        hInstance,  // Instance handle
        NULL        // Additional application data
    );

    if (hwnd == NULL) {
        return 0;
    }

    // Step 3: Display the Window
    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    // Step 4: Run the message loop
    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}