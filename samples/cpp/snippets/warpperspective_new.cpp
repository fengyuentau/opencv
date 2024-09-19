#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

static void getM(float scale, float angle, int w, int h, float *M) {
    float cx = w / 2.f, cy = h / 2.f;

    float MR[6];
    float ca = cosf(angle), sa = sinf(angle);
    MR[0] = scale * ca; MR[1] = scale * sa;     MR[2] = scale * (-cx * ca - cy * sa) + cx;
    MR[3] = -scale * sa; MR[4] = scale * ca;    MR[5] = scale * (cx * sa - cy * ca) + cy;

    /*
        (x0, y0)    (x1, y1)
        /          /
       +----------+
       |          |
       |          |
       +----------+
        \          \
         (x3, y3)   (x2, y2)

        | x'|        | x |
        | y'| = MR * | y |
        | 1 |        | 1 |

        x' = MR[0]*x + MR[1]*y + MR[2];
        y' = MR[3]*x + MR[4]*y + MR[5];
    */
    int x = 0, y = 0;
    float x0 = MR[0] * x + MR[1] * y + MR[2],
          y0 = MR[3] * x + MR[4] * y + MR[5];
    x = 0, y = w;
    float x1 = MR[0] * x + MR[1] * y + MR[2],
          y1 = MR[3] * x + MR[4] * y + MR[5];
    x = h, y = w;
    float x2 = MR[0] * x + MR[1] * y + MR[2],
          y2 = MR[3] * x + MR[4] * y + MR[5];
    x = h, y = 0;
    float x3 = MR[0] * x + MR[1] * y + MR[2],
          y3 = MR[3] * x + MR[4] * y + MR[5];

    float d = scale * ((w+h)/2.f) * 0.2;
    float speed = 1.f;
    x0 += d * cosf(0);       y0 += d * sinf(0);
    x1 += d * cosf(speed);   y1 += d * sinf(speed);
    x2 += d * cosf(2*speed); y2 += d * sinf(2*speed);
    x3 += d * cosf(3*speed); y3 += d * sinf(3*speed);

    const cv::Point2f src[] = {
        {0, 0},
        {0, (float)w},
        {(float)h, (float)w},
        {(float)h, 0},
    };
    const cv::Point2f dst[] = {
        {x0, y0},
        {x1, y1},
        {x2, y2},
        {x3, y3},
    };
    auto M_ = cv::getPerspectiveTransform(src, dst);
    auto pM_ = M_.ptr<double>();
    for (size_t i = 0; i < M_.total(); i++) {
        M[i] = pM_[i];
    }
}

int main(int argc, char** argv) {
    Mat img = imread(argv[1], 1);

    if (argc > 2) {
        std::string type_str = argv[2];
        if (type_str == "8uc1") {
            cvtColor(img, img, COLOR_BGR2GRAY);
        } else if (type_str == "8uc4") {
            cvtColor(img, img, COLOR_BGR2BGRA);
        } else if (type_str == "32fc3") {
            img.convertTo(img, CV_32FC3);
        } else if (type_str == "32fc1") {
            cvtColor(img, img, COLOR_BGR2GRAY);
            img.convertTo(img, CV_32FC1);
        } else if (type_str == "32fc4") {
            cvtColor(img, img, COLOR_BGR2BGRA);
            img.convertTo(img, CV_32FC4);
        } else if (type_str == "16uc3") {
            img.convertTo(img, CV_16UC3);
        } else if (type_str == "16uc1") {
            cvtColor(img, img, COLOR_BGR2GRAY);
            img.convertTo(img, CV_16UC1);
        } else if (type_str == "16uc4") {
            cvtColor(img, img, COLOR_BGR2BGRA);
            img.convertTo(img, CV_16UC4);
        }
    }

    Mat canvas0(img.size(), img.type());
    Mat canvas1(img.size(), img.type());
    int iangle = -1;
    int borderType = BORDER_CONSTANT;
    Scalar borderValue(0, 128, 0, 0);

    for (;;) {
        iangle = (iangle + 1) % (360*4);
        float angle = iangle*CV_PI/180.f*0.25f;
        float scale = 1 + 0.2f*sin(angle);
        float M[9];
        getM(scale, angle, img.cols, img.rows, M);

        double t0 = getTickCount();
        warpPerspective(img, canvas0, Mat(3, 3, CV_32F, M), canvas0.size(), INTER_LINEAR, borderType, borderValue);
        t0 = getTickCount() - t0;

        // Call new warpPerspective
        double t1 = getTickCount();
        warpPerspective(img, canvas1, Mat(3, 3, CV_32F, M), canvas1.size(), INTER_LINEAR, borderType, borderValue, cv::ALGO_HINT_APPROX);
        t1 = getTickCount() - t1;

        printf("opencv time = %.1fms, new time = %.1fms\n", t0*1000./getTickFrequency(), t1*1000./getTickFrequency());
        imshow("result (opencv)", canvas0);
        imshow("result (new)", canvas1);
        int c = waitKey(1);
        if (c < 0)
            continue;
        if ((c & 255) == 27)
            break;
        if ((waitKey() & 255) == 27)
            break;
    }
}
