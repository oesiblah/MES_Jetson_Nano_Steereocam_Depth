#include <iostream>
#include <ctime>
#include <iomanip>
#include <sys/time.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

#include "TaraXL.h"
#include "TaraXLCam.h"
#include "TaraXLDepth.h"

#include "detectNet.h"
#include <signal.h>
#include "cudaRGB.h"

using namespace std;
using namespace cv;
using namespace TaraXLSDK;

int main (int argc, char** argv) {

    commandLine cmdLine(argc, argv, (const char*)NULL);
    TaraXL taraxlCam;
    TaraXLCamList taraxlCamList;
    ResolutionList supportedResolutions;
    ACCURACY_MODE selectedMode;

    uint iAccuracyMode;
    TARAXL_STATUS_CODE status;

    detectNet* net = detectNet::Create(cmdLine);
    if(!net)
    {
        LogError("detectNet: failed to load detectNet model\n");
        return 0;
    }

    const uint32_t overlayFlags = detectNet::OverlayFlagsFromStr("box,labels,conf");

    status = taraxlCam.enumerateDevices(taraxlCamList);
    if (status != TARAXL_SUCCESS) {

        cout << "Camera enumeration failed" << endl;
        return 1;
    }

    if (taraxlCamList.size() == 0) {

        cout << "No cameras connected" << endl;
        return 1;
    }
    cout << endl << "Select a Accuracy mode:" << endl;
    cout << "0: High Accuracy" << endl <<"1: Low Accuracy "<<endl<<"2: Ultra Accuracy" << endl;
    cin >> iAccuracyMode;

    if (cin.fail()) {

        cout << "Invalid input" << endl;
        return 1;
    }

    if (iAccuracyMode == 0) {

        selectedMode = HIGH;
    }
    else if (iAccuracyMode == 1) {

        selectedMode = LOW;
    }
    else if (iAccuracyMode == 2) {

        selectedMode = ULTRA;
    }

    else {

        cout << "Invalid input" << endl;
        return 1;
    }

    vector<Ptr<TaraXLDepth> > taraxlDepthList;
    vector<Mat> left, right, grayDisp, colorDisp, depthMap;
    vector<string> cameraUniqueIdList;
    Resolution res;
    for(int i = 0 ; i < taraxlCamList.size() ; i++)
    {
        status = taraxlCamList.at(i).connect();
        if (status != TARAXL_SUCCESS) {

            cout << "Camera connect failed " << status << endl;
            return 1;
        }
        Ptr<TaraXLDepth> depth;
        cout << "Camera connect status" << status << endl;
        depth = new TaraXLDepth(taraxlCamList.at(i));
        if (depth == NULL)
        {
            cout << "Unable to create instance to TaraDepth" << endl;
            return 1;
        }
        depth->setAccuracy(selectedMode);
        taraxlDepthList.push_back(depth);
            string id;
            taraxlCamList.at(i).getCameraUniqueId(id);
        cameraUniqueIdList.push_back(id);
        string windowName = "CAMERA : "+ id;
        namedWindow(windowName, CV_WINDOW_AUTOSIZE);

        Mat sample;
        left.push_back(sample);
        right.push_back(sample);
        grayDisp.push_back(sample);
        colorDisp.push_back(sample);
        depthMap.push_back(sample);
        taraxlCamList.at(i).getResolution(res);
    }
    
    while(1)
    {
        for(int i = 0 ; i < taraxlCamList.size() ; i++)
        {
            status = taraxlDepthList.at(i)->getMap(left.at(i), right.at(i), grayDisp.at(i), true, depthMap.at(i), true, TARAXL_DEFAULT_FILTER);
            if (status != TARAXL_SUCCESS)
            {
                cout << "Get map failed" << endl;
                delete taraxlDepthList.at(i);
                return 1;
            }
            grayDisp.at(i).convertTo(grayDisp.at(i),CV_8U);

            detectNet::Detection* detections = NULL;
            Mat left_color;
            cvtColor(left.at(i), left_color, COLOR_GRAY2BGR);

            uchar3* imgBufferRGB = NULL;
            float4* imgBufferRGBAf = NULL;
            cudaMalloc((void**)&imgBufferRGB, left_color.cols * sizeof(uchar3) * left_color.rows);
            cudaMalloc((void**)&imgBufferRGBAf, left_color.cols * sizeof(float4) * left_color.rows);
            cudaMemcpy2D((void*)imgBufferRGB, left_color.cols*sizeof(uchar3), (void*)left_color.data, left_color.step, left_color.cols*sizeof(uchar3), left_color.rows, cudaMemcpyHostToDevice);
            cudaRGB8ToRGBA32(imgBufferRGB, imgBufferRGBAf, left_color.cols, left_color.rows);

            const int numDetections = net->Detect((float*)imgBufferRGBAf, (uint32_t)left_color.cols, (uint32_t)left_color.rows, &detections, overlayFlags);

            cv::cvtColor(left.at(i), left.at(i), COLOR_GRAY2BGR);
            if(numDetections > 0)
            {
                for(int n=0; n < numDetections; n++)
                {
                    LogVerbose("detected obj %i class #%u (%s) confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
                    LogVerbose("bounding box %i  (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height());
                    Point p1(detections[n].Left + detections[n].Width(), detections[n].Top + detections[n].Height());
                    Point p2(detections[n].Left, detections[n].Top);
                    Scalar magenta = Scalar(255,10,255);
                    Point center(detections[n].Left + detections[n].Width()*0.5, detections[n].Top + detections[n].Height()*0.5);
                    putText(left.at(i), net->GetClassDesc(detections[n].ClassID), Point(detections[n].Left + detections[n].Width()*0.5, (detections[n].Top + 25)), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,200,200), 2);
                    rectangle(left.at(i), p1, p2, magenta, 1, 8, 0);

                    Mat disp, disp_32;
                    float DepthValue;
                    Rect recROI(detections[n].Left + detections[n].Width()*0.5, detections[n].Top + detections[n].Height()*0.5, 1, 1);
                    disp = depthMap.at(i)(recROI);
                    disp.convertTo(disp_32, CV_32FC1, 1.0);
                    Scalar MeanDisp = mean(disp_32);
                    DepthValue = (float)MeanDisp.val[0];
                    String test;
                    test = "Distance: " + std::to_string(DepthValue) + "cm";
                    putText(left.at(i), test, Point(detections[n].Left + detections[n].Width()*0.5, (detections[n].Top + 50)), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,200,200), 2);
                    std::cout << "Object " << n << " distance: " << DepthValue << "cm" << std::endl;
                }
            }

            string windowName = "CAMERA : "+ cameraUniqueIdList.at(i);
            imshow(windowName, left.at(i));
            cudaFree(imgBufferRGB);
            cudaFree(imgBufferRGBAf);
        }
        int keycode = waitKey(30) & 0xff;
        if(keycode == 27) break;
    }
    exit(0);
}
