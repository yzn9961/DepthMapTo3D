#include <opencv2/opencv.hpp>
//#include<opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;
using namespace cv;

//string left_file = "../StereoImage/left/L_10.jpg";
//string right_file = "../StereoImage/right/R_10.jpg";
string parameterFile = "../StereoImage/my2cam.yaml";
string Q_Path = "../StereoImage/extrinsics.yml";


cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r,Q;
int rows_l;
int cols_l;
int rows_r;
int cols_r;
typedef Matrix<double,6,1> Vector6d;

void Getparameter()
{
    cv::FileStorage fsSettings(parameterFile, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
    }

    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    rows_l = fsSettings["LEFT.height"];
    cols_l = fsSettings["LEFT.width"];
    rows_r = fsSettings["RIGHT.height"];
    cols_r = fsSettings["RIGHT.width"];

    fsSettings.open(Q_Path,cv::FileStorage::READ);
    fsSettings["Q"] >> Q;
    if(Q.empty() || K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
            rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
    }
}
Mat show_stereoCalib(Mat rectifyImageL,Mat rectifyImageR) //展示双目立体标定结果
{
    Mat canvas;
    //cvtColor(canvas,canvas,COLOR_BGR2GRAY);
    double sf;
    int w, h;
    sf = 600. / MAX(cv::Size(cols_l,rows_l).width, cv::Size(cols_l,rows_l).height);
    w = cvRound(cv::Size(cols_l,rows_l).width * sf);
    h = cvRound(cv::Size(cols_l,rows_l).height * sf);
    canvas.create(h, w * 2, CV_8UC3);
    //左图像画到画布上
    Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分
    resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小
    //Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域
        //cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
    //rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形
    //cout << "Painted ImageL" << endl;

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分
    resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    //Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
        //cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
    //rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
    //cout << "Painted ImageR" << endl;

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
    //imshow("rectified", canvas);
    return canvas;
}
//点云绘制
void showPointCloud(
    const vector<Vector6d, Eigen::aligned_allocator<Vector6d>>&pointcloud);
int main(int argc,char**argv)
{   if(argc != 2)
    {
        cerr << endl << "ERROR:格式错误！请在可执行文件后 输入双目图像路径！" << endl;
        return 1;
    }
    //// 读取相机参数 ////
    //double fx,fy,cx,cy;
    //double b;
    //读取畸变矫正参数
    Getparameter();
    // 计算畸变矫正参数
    cv::Mat M1l,M2l,M1r,M2r;
    cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
    cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
    //// 读取图像 ////
    Mat left,right;
    left.create(480, 640, CV_8UC3);
    Rect Lrect(0,0, 1280/2,480);//左ROI
    Rect Rrect(1280 / 2, 0, 1280/2, 480); //右ROI
    Mat raw = imread(argv[1],1);
    cv::Mat left_src = raw(Lrect);//读取灰度图像 格式为CV_8UC1: 8深度 无符号0~255 Channel=1
    cv::Mat right_src = raw(Rrect);//CV_8UC1
    
    /// 畸变校正 ///
    Mat rectifyImageL,rectifyImageR,rectifyImageL_gray,rectifyImageR_gray;
    remap(left_src, rectifyImageL, M1l,M2l, INTER_LINEAR);
    remap(right_src, rectifyImageR, M1r,M2r, INTER_LINEAR);
    //imshow("Left",rectifyImageL);
    imshow("Stereo Calibra",show_stereoCalib(rectifyImageL,rectifyImageR));
    cvtColor(rectifyImageL,rectifyImageL_gray,COLOR_BGR2GRAY);
    cvtColor(rectifyImageR,rectifyImageR_gray,COLOR_BGRA2GRAY);
    //// 计算深度图 ////
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32); 
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(rectifyImageL_gray, rectifyImageR_gray, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F,1.0 / 16.0f);
    //cv::imshow("Depth Map origin",disparity_sgbm);//深度图像
    Mat goodlook;// good to look
    divide(disparity, 96.0, goodlook);// (disparity / 96.0)
    cv::imshow("Depth Map convert",goodlook);//便于显示的深度图样式
    //imwrite("Depth.png",disparity / 96.0);

    //// 三维重建 ////
    Mat Point3D;
    cv::reprojectImageTo3D(disparity,Point3D,Q,true,-1);//计算三维空间点坐标
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>>pointcloud;//点云集
    cout << "准备存储点云"<< endl;
    for (int v = 0; v < 480; v++)//行
           for (int u = 0; u < 640; u++) {//列
               if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;//将误匹配点省略
               /*
               Vector6f point(Point3D.at<Vec3f>(v,u)[0],
                       Point3D.at<Vec3f>(v,u)[1],
                       Point3D.at<Vec3f>(v,u)[2],
                       rectifyImageL.at<Vec3b>(v, u)[0],rectifyImageL.at<Vec3b>(v,u)[0]); // 前三维为xyz,第四维为颜色，归一化颜色值
                       */
              // cout<<"正在处理第("<<v<<","<<u<<")位置的像素！";
               Vector6d point(6);
               point << (double)Point3D.at<Vec3f>(v,u)[0], //X
                        (double)Point3D.at<Vec3f>(v,u)[1], //Y
                        (double)Point3D.at<Vec3f>(v,u)[2], //Z
                        rectifyImageL.data[v * rectifyImageL.step + u * rectifyImageL.channels()+2], //R
                        rectifyImageL.data[v * rectifyImageL.step + u * rectifyImageL.channels() + 1], //G
                        rectifyImageL.data[v * rectifyImageL.step + u * rectifyImageL.channels()]; //B

               pointcloud.push_back(point);//将点-颜色存储到容器中
           }
    imshow("src",left_src);
    cv::waitKey(0);
    cout << "有效点云数量为："<<pointcloud.size()<<endl;
    showPointCloud(pointcloud);
}

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>>&pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);//创建窗口并确定尺寸
    glEnable(GL_DEPTH_TEST);//3D可视化时开启，只绘制朝向摄像头一侧的图像
    glEnable(GL_BLEND);//启用颜色混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//颜色混合方式
    //创建一个相机的观察视角
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),//相机视角的尺寸，内参，最近和最远可视距离
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)//设置相机的外参：相机位置，相机朝向(俯仰，左右)，相机机轴方向(相机平面的旋转)——>（0.0, -1.0, 0.0）代表-y方向
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()//创建视图
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)//视图在视窗中的范围，以及视图的长宽比
        .SetHandler(new pangolin::Handler3D(s_cam));//显示s_cam所拍摄的内容

    while (pangolin::ShouldQuit() == false) {//若不关闭openGL窗口
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//清空颜色和深度缓存，防止前后帧之间存在干扰

        d_cam.Activate(s_cam);//激活显示并设置状态矩阵
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);//刷新缓冲区颜色，防止帧间干扰
        glPointSize(2);//所绘点的大小
        /////// 真正的绘图部分 //////////
        glBegin(GL_POINTS);//点设置的开始
        for (auto &p: pointcloud) {//auto根据后面的变量值自行判断变量类型，继承点云
            glColor3d(p[3]/255.0, p[4]/255.0, p[5]/255.0);//RGB三分量相等即是灰度图像
            glVertex3d(p[0], p[1], p[2]);//确定点坐标
        }
        glEnd();//点设置的结束
        ///////////////////////////////
        pangolin::FinishFrame();//开始执行后期渲染，事件处理以及帧交换
        usleep(5000);   // sleep 5 ms
    }
    return;
}



