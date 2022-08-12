/*
 * @Description: FPFH.cpp
 * @version: 1.0.0.0
 * @Author: Haydn<haydn@cz-robots.com>
 * @Date: 8/10/22
*/
#include "vector"
#include <pcl/console/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/thread/thread.hpp>
#include <pcl/features/fpfh_omp.h> //包含fpfh加速计算的omp(多核并行计算)
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_features.h> //特征的错误对应关系去除
#include <pcl/registration/correspondence_rejection_sample_consensus.h> //随机采样一致性去除
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/features/integral_image_normal.h>
#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/statistical_outlier_removal.h>
// 包含相关头文件

// This function displays the help
void showHelp(char *program_name) {
    std::cout << std::endl;
    std::cout << "Usage: " << program_name << " cloud_filename.[pcd|ply]" << std::endl;
    std::cout << "-h or --help :  Show  help." << std::endl;
}

bool next_iteration = false;

void
keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                      void *nothing) {
    if (event.getKeySym() == "space" && event.keyDown())
        next_iteration = true;
}

using namespace std;
typedef pcl::PointCloud<pcl::PointXYZ> pointcloud;
typedef pcl::PointCloud<pcl::Normal> pointnormal;
typedef pcl::PointCloud<pcl::FPFHSignature33> fpfhFeature;

// This is the main function
int main(int argc, char **argv) {
    pcl::console::TicToc time;
    time.tic();
    // Show help
    if (pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help")) {
        std::cout << "没有help." << std::endl;
        return 0;
    }

    // Load file | Works with PCD and PLY files
    pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile(argv[1], *target) < 0) {
        std::cout << "Error loading point target " << argv[0] << std::endl << std::endl;
        showHelp(argv[0]);
        return -1;
    }
    if (pcl::io::loadPCDFile(argv[2], *source) < 0) {
        std::cout << "Error loading point source " << argv[2] << std::endl << std::endl;
        showHelp(argv[0]);
        return -1;
    }

//    Eigen::Matrix4d target_trans, source_trans;
//    target_trans <<  0.9393691472558041,   -0.3428872573638045, -0.003732805692202962,     55.13104448487393,
//    0.342488287673515  ,  0.9376267945201996 ,  0.05964701936187008 ,    8.032505612959786,
//     -0.01695222425734497,  -0.05730901194244873,    0.9982125521352951 ,   0.5193597494147779,
//    0            ,         0,                     0  ,                   1;
//
//    source_trans <<  -0.9457604460984244 ,  -0.3248267822841094, -0.004975246922563866  ,   55.71692423611487,
//    0.3235248368097304  , -0.9431383801969908  , 0.07629999164107412   ,  9.094755525355399,
//    -0.02947662698183055 ,  0.07055189769341771  ,  0.9970724939607719  ,  0.4732753888274806,
//    0                ,     0                 ,    0                   ,  1;


//    pcl::transformPointCloud(*target, *target, target_trans);
//    pcl::transformPointCloud(*source, *source, source_trans);

    std::cout << "source size is: " << source->size() << std::endl;
    std::cout << "target size is: " << target->size() << std::endl;

    //desturb
//    float theta = M_PI / 4; // The angle of rotation in radians
//    Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
    // Define a translation of 2.5 meters on the x axis.
//    transform_2.translation() << 2, 0.0, 0.0;
    // The same rotation matrix as before; theta radians around Z axis
//    transform_2.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()));

    //remove outlier
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_inliers(new pcl::PointCloud<pcl::PointXYZ>);
    // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(source);  //设置输入
    sor.setMeanK(50);  //设置用于平均距离估计的 KD-tree最近邻搜索点的个数.
    sor.setStddevMulThresh(1.0); //高斯分布标准差的倍数, 也就是 u+1*sigma,u+2*sigma,u+3*sigma 中的 倍数1、2、3
    sor.filter(*source); // 滤波后输出

    sor.setInputCloud(target);  //设置输入
    sor.setMeanK(50);  //设置用于平均距离估计的 KD-tree最近邻搜索点的个数.
    sor.setStddevMulThresh(1.0); //高斯分布标准差的倍数, 也就是 u+1*sigma,u+2*sigma,u+3*sigma 中的 倍数1、2、3
    sor.filter(*target); // 滤波后输出



    //体素化
    //pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_grid;
    pcl::VoxelGrid<pcl::PointXYZ> approximate_voxel_grid;
    approximate_voxel_grid.setLeafSize(0.2, 0.2, 0.2); //网格边长.这里的数值越大，则精简的越厉害（剩下的数据少）
    //pointcloud::Ptr source(new pointcloud);
    //1
    pointcloud::Ptr sample_source(new pointcloud);
    approximate_voxel_grid.setInputCloud(source);
    approximate_voxel_grid.filter(*sample_source);
    cout << "voxel grid  Filte source size is: " << sample_source->size() << endl;
    pcl::copyPointCloud(*sample_source, *source);
    //2
    approximate_voxel_grid.setInputCloud(target);
    approximate_voxel_grid.filter(*sample_source);
    cout << "voxel grid  Filte target size is: " << sample_source->size() << endl;
    pcl::copyPointCloud(*sample_source, *target);

    //可视化
    pcl::visualization::PCLVisualizer view("111");
    int v1, v2, v3;

    view.createViewPort(0, 0.0, 0.3, 1.0, v1);
    view.createViewPort(0.3, 0.0, 0.6, 1.0, v2);
    view.createViewPort(0.6, 0.0, 1., 1.0, v3);

    // The color we will be using
    float bckgr_gray_level = 0.0;  // Black
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    view.setBackgroundColor(0, 0, 0, v1);
    view.setBackgroundColor(0.05, 0, 0, v2);
    //source: greed  target: white
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> sources_cloud_color(source, 0, 250, 0);
    view.addPointCloud(source, sources_cloud_color, "sources_cloud_v1", v1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_cloud_color(target, (int) 255 * txt_gray_lvl,
                                                                                       (int) 255 * txt_gray_lvl,
                                                                                       (int) 255 * txt_gray_lvl);
    view.addPointCloud(target, target_cloud_color, "target_cloud_v1", v1);
    view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sources_cloud_v1");
    //align: red
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligend_cloud_color(source, 255, 0, 0);
    view.addPointCloud(source, aligend_cloud_color, "aligend_cloud_v2", v2);
    view.addPointCloud(target, target_cloud_color, "target_cloud_v2", v2);
    view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "aligend_cloud_v2");
    view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud_v2");


    view.addPointCloud(source, sources_cloud_color, "guess_cloud_v3", v3);
    view.addPointCloud(target, target_cloud_color, "target_cloud_v3", v3);
    view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "guess_cloud_v3");
    view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud_v3");


    view.addText ("fitnessscore", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2", v2);

    // Register keyboard callback :
    view.registerKeyboardCallback(&keyboardEventOccurred, (void *) NULL);

    //brute force filter
    float x_range = 2.;
    float translation_step = 1;
    float yaw_range = 180.;
    float angle_step = 30.;
    double min_score = 10000000;
    Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
    //init vector
    std::vector<Eigen::Affine3f> guess_table;
    for (float x = -x_range; x < x_range;) {
        x = x + translation_step;
        for (float y = -x_range; y < x_range;) {
            y = y + translation_step;
            for (float angle = -yaw_range; angle < yaw_range;) {
                angle = angle + angle_step;
                float theta = angle * M_PI / 180.; // The angle of rotation in radians
                Eigen::Affine3f guess = Eigen::Affine3f::Identity();
                guess.translation() << x, y, 0.0;
                guess.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()));
                guess_table.push_back(guess);
                std::cout << "x/y/angle: " << x << " " << y << " " << angle << std::endl;
            }
        }
    }

    int index = 0;
    while (!view.wasStopped() && index < guess_table.size()) {
        view.spinOnce();
        //The user press "space"
        if (next_iteration) {
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setMaxCorrespondenceDistance(2);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(0.0000000001);
            icp.setEuclideanFitnessEpsilon(0.0000000001);
            icp.setRANSACIterations(0);
            icp.setInputSource(source);
            icp.setInputTarget(target);
            auto guess = guess_table[index];
            std::cout << "guess: " << guess.matrix() << std::endl;
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(
                    new pcl::PointCloud<pcl::PointXYZ>);  // ICP output point cloud
            icp.align(*cloud_icp, guess.matrix());
            view.updatePointCloud(cloud_icp, aligend_cloud_color, "aligend_cloud_v2");
            pcl::PointCloud<pcl::PointXYZ>::Ptr guess_point(
                    new pcl::PointCloud<pcl::PointXYZ>);  // ICP output point cloud
            pcl::transformPointCloud(*source, *guess_point, guess);
            view.updatePointCloud(guess_point, sources_cloud_color, "guess_cloud_v3");
            index++;
            std::string fitness = "fitness score: " + to_string(icp.getFitnessScore());
            view.updateText (fitness, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2");
        }
        next_iteration = false;
    }

    std::cout << "match result: " << result << std::endl;
    std::cout << "score result: " << min_score << std::endl;

    std::cout << "计算结束" << std::endl << std::endl;
    cout << "代码段运行时间: " << time.toc() / 1000 << "s" << endl;//计算程序运行时间

//    pcl::PointCloud<pcl::PointXYZ>::Ptr comp_cloud(new pcl::PointCloud<pcl::PointXYZ>());


    //pcl::io::savePCDFile("crou_output.pcd", *align);

    cout << "代码段运行时间: " << time.toc() / 1000 << "s" << endl;//计算程序运行时间
    while (!view.wasStopped()) { // Display the visualiser until 'q' key is pressed
        view.spinOnce();
    }

    //system("pause");//不注释，命令行会提示按任意键继续，注释会直接跳出
    return 0;
}

