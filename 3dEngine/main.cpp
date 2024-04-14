
#include <render.h>

#define IDT_TIMER1 1
float cameraX = 0.0f;;
float cameraY = 0.0f; 
float cameraZ = 0.0f; 
std::string cameraName = "MainCamera"; 
float fov = 60.0f; //
float aspectRatio = 16.0f / 9.0f; 
float nearPlane = 0.1f; // Near clipping plane
float farPlane = 100.0f; // Far clipping plane
camera cam(cameraX, cameraY, cameraZ, cameraName, fov, aspectRatio, nearPlane, farPlane);
mesh* dolphin = loadOBJ("C:\\Users\\boden\\Downloads\\10014_dolphin_v2_max2011_it2.obj");
mesh* car = loadOBJ("C:\\Users\\Boden\\Downloads\\uploads_files_2792345_Koenigsegg.obj");
mesh* dolphin1 = loadOBJ("C:\\Users\\boden\\Downloads\\10014_dolphin_v2_max2011_it2.obj");
mesh* dolphin2 = loadOBJ("C:\\Users\\boden\\Downloads\\10014_dolphin_v2_max2011_it2.obj");

world World;

double fpsAverage = 0;
int count;
ThreadPool* pool;

//mesh cube = CreateCube(0, 0, 0, 4.0);


LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE:
        pool = createThreadPool(std::thread::hardware_concurrency());
        //pool = createThreadPool(1);
        dolphin->Name = "dolphin";
        dolphin1->Name = "dolphin2";
        dolphin2->Name = "OG2";
        car->Name = "car";
        car->setBatchSize(car->vertexList.size());
        dolphin->setBatchSize(dolphin->vertexList.size());
        World.setThreadPool(pool);
        World.setCam(cam);
        dolphin = World.addMesh(dolphin);
        car = World.addMesh(car);
        dolphin->rotate(90, 0, 0);
        car->rotate(0, 0, 180);
        car->transform(0, 5, 25);
        World.deRenderObject("car");





        





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
        cam.aspectRatio = width / height;
        cam.moveCam( 0.0f);
        cam.rotateCam(3);
        HDC hdesktop = GetDC(0);
        HDC memdc = CreateCompatibleDC(hdesktop);
       
        HBITMAP hbitmap = CreateCompatibleBitmap(hdesktop, width, height);
        
        gmtl::Vec4f cen = dolphin->Center();
        SelectObject(memdc, hbitmap);
        // Fill the bitmap with red color
        
        COLORREF redColor = RGB(255, 0, 0);

        //World.worldObjects.at("OG").rotate(0, 0, 2);
        dolphin->rotate(0, 1, 1);
        dolphin->transform(0, 0, 0.03);




        

        World.renderWorld(memdc, redColor, width, height);
        //fps calculation
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        std::wstringstream ss;
        std::wstringstream x;
        std::wstringstream y;
        std::wstringstream z;
        std::wstringstream avr;

        x << L"X: " << cen[0];
        y << L"Y: " << cen[1];
        z << L"Z: " << cen[2];

        ss << L"FPS: " << 1e+9f/duration.count();
        fpsAverage += 1e+9f / duration.count();
        count++;
        avr << "FPS: " << fpsAverage / count;

        std::wstring FPSStr = ss.str();
        std::wstring X = x.str();
        std::wstring Y = y.str();
        std::wstring Z = z.str();
        std::wstring avR = avr.str();


        SetTextColor(memdc, RGB(255, 255, 255));
        SetBkMode(memdc, TRANSPARENT);

        TextOut(memdc, 50, 50, FPSStr.c_str(), 13);
        TextOut(memdc, 50, 100, X.c_str(), 13);
        TextOut(memdc, 50, 115, Y.c_str(), 13);
        TextOut(memdc, 50, 130, Z.c_str(), 13);
        TextOut(memdc, 50, 145, avR.c_str(), 13);

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