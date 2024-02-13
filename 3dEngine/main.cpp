
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
mesh cube;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE:
        rotate(cube, 180.0f, 90.0f, 0.0f);

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
        float width;
        float height;
        HDC hdc = BeginPaint(hwnd, &ps);
        if (GetWindowRect(hwnd, &rect))
        {
            width = rect.right - rect.left;
            height = rect.bottom - rect.top;
        }
        else {
            break;
        }
        /*point p1(0, 0, 0);
        point p2(0,0.5,0);
        point p3(0.5,0,0);
        triangle test(p1, p2, p3);
        mesh Mesh;
        Mesh.vertexList.push_back(test);*/
        cam.aspectRatio = width / height;
        moveCam(cam, 1.0f);
        transform(cube, 0.0f, 0.0f, 0.0f);
 
        HDC hdesktop = GetDC(0);
        HDC memdc = CreateCompatibleDC(hdesktop);
       
        HBITMAP hbitmap = CreateCompatibleBitmap(hdesktop, width, height);
        
        gmtl::Vec4f cen =  Center(cube);
        SelectObject(memdc, hbitmap);
        // Fill the bitmap with red color
        
        COLORREF redColor = RGB(255, 0, 0);
        DrawMesh(memdc, cube, redColor, width, height, cam);
        rotate(cube, 0.0f, 1.0f, 0.0f);

        //fps calculation
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        std::wstringstream ss;
        std::wstringstream x;
        std::wstringstream y;
        std::wstringstream z;

        x << L"X: " << cen[0];
        y << L"Y: " << cen[1];
        z << L"Z: " << cen[2];

        ss << L"FPS: " << 1e+9/duration.count();
        std::wstring FPSStr = ss.str();
        std::wstring X = x.str();
        std::wstring Y = y.str();
        std::wstring Z = z.str();
        SetTextColor(memdc, RGB(255, 255, 255));
        SetBkMode(memdc, TRANSPARENT);

        TextOut(memdc, 50, 50, FPSStr.c_str(), 13);
        TextOut(memdc, 50, 100, X.c_str(), 13);
        TextOut(memdc, 50, 115, Y.c_str(), 13);
        TextOut(memdc, 50, 130, Z.c_str(), 13);

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