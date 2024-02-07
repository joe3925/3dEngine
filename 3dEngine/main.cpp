
#include "render.hpp"
// Define a window procedure function to handle messages

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        std::vector<point> pointlist;
        pointlist.push_back(point(0.0f, 0.0f, 0.0f));
        pointlist.push_back(point(1.0f, 0.0f, 0.0f));
        pointlist.push_back(point(1.0f, 1.0f, 0.0f));
        pointlist.push_back(point(0.0f, 1.0f, 0.0f));
        pointlist.push_back(point(0.0f, 0.0f, 1.0f));
        pointlist.push_back(point(1.0f, 0.0f, 1.0f));
        pointlist.push_back(point(1.0f, 1.0f, 1.0f));
        pointlist.push_back(point(0.0f, 1.0f, 1.0f));
        pointlist.push_back(point(0.5f, 0.5f, 0.0f));
        pointlist.push_back(point(0.5f, 0.5f, 1.0f));
        mesh Mesh;
        Mesh.vertexList.push_back(triangle(pointlist[0], pointlist[4], pointlist[5]));
        Mesh.vertexList.push_back(triangle(pointlist[0], pointlist[5], pointlist[1]));
        Mesh.vertexList.push_back(triangle(pointlist[1], pointlist[5], pointlist[6]));
        Mesh.vertexList.push_back(triangle(pointlist[1], pointlist[6], pointlist[2]));
        Mesh.vertexList.push_back(triangle(pointlist[2], pointlist[6], pointlist[7]));
        Mesh.vertexList.push_back(triangle(pointlist[2], pointlist[7], pointlist[3]));
        Mesh.vertexList.push_back(triangle(pointlist[3], pointlist[7], pointlist[4]));
        Mesh.vertexList.push_back(triangle(pointlist[3], pointlist[4], pointlist[0]));
        Mesh.vertexList.push_back(triangle(pointlist[8], pointlist[5], pointlist[4]));
        Mesh.vertexList.push_back(triangle(pointlist[8], pointlist[6], pointlist[5]));
        Mesh.vertexList.push_back(triangle(pointlist[8], pointlist[7], pointlist[6]));
        Mesh.vertexList.push_back(triangle(pointlist[8], pointlist[4], pointlist[7]));
        Mesh.vertexList.push_back(triangle(pointlist[9], pointlist[5], pointlist[4]));
        Mesh.vertexList.push_back(triangle(pointlist[9], pointlist[6], pointlist[5]));
        Mesh.vertexList.push_back(triangle(pointlist[9], pointlist[7], pointlist[6]));
        Mesh.vertexList.push_back(triangle(pointlist[9], pointlist[4], pointlist[7]));
        transform(Mesh, 0, 0, 2);
        RECT rect;
        int width;
        int height;
        HDC hdc = BeginPaint(hwnd, &ps);
        camera Camera(0, 1, 0, "cam", 90);
        Camera.Postition[1];
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
        DrawMesh(memdc, Mesh, redColor, width, height);

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