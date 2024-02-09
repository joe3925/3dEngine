
#include "render.hpp"
#define IDT_TIMER1 1
float cameraX = 0.0f;;
float cameraY = 0.0f; 
float cameraZ = 0.0f; 
std::string cameraName = "MainCamera"; 
float fov = 90.0f; //
float aspectRatio = 16.0f / 9.0f; 
float nearPlane = 0.1f; // Near clipping plane
float farPlane = 100.0f; // Far clipping plane
camera cam(cameraX, cameraY, cameraZ, cameraName, fov, aspectRatio, nearPlane, farPlane);
mesh cube = CreateCube(0.4f, 0.5f, 5.0f, 0.5f);

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE:
        if (SetTimer(hwnd, IDT_TIMER1, 0, NULL) != 0) {
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
        auto start = std::chrono::high_resolution_clock::now();
        PAINTSTRUCT ps;

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
        aspectRatio = width / height;
        moveCam(cam, 0.05f);
        transform(cube, 0, 0, 0.0f);
        rotate(cube, 0, 1.5, 0);

 
        HDC hdesktop = GetDC(0);
        HDC memdc = CreateCompatibleDC(hdesktop);
       
        HBITMAP hbitmap = CreateCompatibleBitmap(hdesktop, width, height);
        
        
        SelectObject(memdc, hbitmap);
        // Fill the bitmap with red color
        
        COLORREF redColor = RGB(255, 0, 0);
        DrawMesh(memdc, cube, redColor, width, height, cam);
        //fps calculation
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        std::wstringstream ss;
        ss << L"FPS: " << 1e+9/duration.count();
        std::wstring FPSStr = ss.str();
        SetTextColor(memdc, RGB(255, 255, 255));
        SetBkMode(memdc, TRANSPARENT);
        TextOut(memdc, 50, 50, FPSStr.c_str(), 13);
        //apply frame
        BitBlt(hdc, 0, 0, width, height, memdc, 0, 0, SRCCOPY);
        //clean up
        DeleteObject(hbitmap); 
        DeleteDC(memdc);
        ReleaseDC(0, hdesktop); 
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