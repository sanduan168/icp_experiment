/*
 * @Description: FPFH.cpp
 * @version: 1.0.0.0
 * @Author: Haydn<haydn@cz-robots.com>
 * @Date: 8/10/22
*/
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
// 包含相关头文件

// This function displays the help
void showHelp(char *program_name) {
    std::cout << std::endl;
    std::cout << "Usage: " << program_name << " cloud_filename.[pcd|ply]" << std::endl;
    std::cout << "-h or --help :  Show  help." << std::endl;
}

using namespace std;
typedef pcl::PointCloud<pcl::PointXYZ> pointcloud;
typedef pcl::PointCloud<pcl::Normal> pointnormal;
typedef pcl::PointCloud<pcl::FPFHSignature33> fpfhFeature;

fpfhFeature::Ptr compute_fpfh_feature(pointcloud::Ptr input_cloud, pcl::search::KdTree<pcl::PointXYZ>::Ptr tree) {
    //法向量
    pointnormal::Ptr point_normal(new pointnormal);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> est_normal;
    est_normal.setInputCloud(input_cloud);
    est_normal.setSearchMethod(tree);
    est_normal.setKSearch(20);
    est_normal.compute(*point_normal);
    //fpfh 估计
    fpfhFeature::Ptr fpfh(new fpfhFeature);
    //pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> est_target_fpfh;
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> est_fpfh;
    est_fpfh.setNumberOfThreads(8); //指定4核计算
    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree4 (new pcl::search::KdTree<pcl::PointXYZ> ());
    est_fpfh.setInputCloud(input_cloud);
    est_fpfh.setInputNormals(point_normal);
    est_fpfh.setSearchMethod(tree);
    est_fpfh.setKSearch(10);
    est_fpfh.compute(*fpfh);

    return fpfh;

}

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
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile(argv[1], *cloud) < 0) {
        std::cout << "Error loading point cloud " << argv[0] << std::endl << std::endl;
        showHelp(argv[0]);
        return -1;
    }
    if (pcl::io::loadPCDFile(argv[2], *source) < 0) {
        std::cout << "Error loading point cloud " << argv[2] << std::endl << std::endl;
        showHelp(argv[0]);
        return -1;
    }

    Eigen::Matrix4d cloud_trans, source_trans;
    cloud_trans <<  0.9734510039862292,   -0.2288926569271857, -0.001137744542119737,     56.29268290726716,
    0.228789424238976  ,   0.973139225644497 , -0.02560169698055988   ,    8.2628585076479,
    0.006967224284269906 ,  0.02466169371110208  ,  0.9996715753932965  ,  0.6699115478746123,
    0             ,        0    ,                 0       ,              1;

    source_trans <<  -0.005618828254460695  , -0.9999815733000071 , 0.002289613884658512    ,  57.7216876434626,
    0.9999769223133873, -0.005627504781703152 ,-0.003800859440756948   ,  3.299780444328192,
    0.003813674336404089 , 0.002268204639150773   , 0.9999901555192352 ,   0.6290907281753199,
    0            ,         0         ,            0    ,                 1;



//    float theta = M_PI / 4; // The angle of rotation in radians
//    Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
//    // Define a translation of 2.5 meters on the x axis.
//    transform_2.translation() << 2.5, 0.0, 0.0;
//    // The same rotation matrix as before; theta radians around Z axis
//    transform_2.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()));

    pcl::transformPointCloud(*cloud, *cloud, cloud_trans);
    pcl::transformPointCloud(*source, *source, source_trans);


    std::cout << "source size is: " << source->size() << std::endl;
    std::cout << "cloud size is: " << cloud->size() << std::endl;

    //体素化
    //pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_grid;
    pcl::VoxelGrid<pcl::PointXYZ> approximate_voxel_grid;
    approximate_voxel_grid.setLeafSize(0.8, 0.8, 0.8); //网格边长.这里的数值越大，则精简的越厉害（剩下的数据少）
    //pointcloud::Ptr source(new pointcloud);
    //1
    pointcloud::Ptr sample_source(new pointcloud);
    approximate_voxel_grid.setInputCloud(source);
    approximate_voxel_grid.filter(*sample_source);
    cout << "voxel grid  Filte cloud size is: " << sample_source->size() << endl;
    pcl::copyPointCloud(*sample_source, *source);
    //2
    approximate_voxel_grid.setInputCloud(cloud);
    approximate_voxel_grid.filter(*sample_source);
    cout << "voxel grid  Filte cloud size is: " << sample_source->size() << endl;
    pcl::copyPointCloud(*sample_source, *cloud);


    //
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    fpfhFeature::Ptr source_fpfh = compute_fpfh_feature(source, tree);
    fpfhFeature::Ptr cloud_fpfh = compute_fpfh_feature(cloud, tree);
    std::cout << "对齐开始" << std::endl << std::endl;
    //对齐(占用了大部分运行时间)
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
    sac_ia.setInputSource(source);
    sac_ia.setSourceFeatures(source_fpfh);
    sac_ia.setInputTarget(cloud);
    sac_ia.setTargetFeatures(cloud_fpfh);
    pointcloud::Ptr align(new pointcloud);
    sac_ia.setNumberOfSamples(20);  //设置每次迭代计算中使用的样本数量（可省）,可节省时间
    sac_ia.setCorrespondenceRandomness(10); //设置计算协方差时选择多少近邻点，该值越大，协防差越精确，但是计算效率越低.(可省)
    sac_ia.align(*align);

    std::cout << "计算结束" << std::endl << std::endl;
    cout << "代码段运行时间: " << time.toc() / 1000 << "s" << endl;//计算程序运行时间
    //可视化
    pcl::visualization::PCLVisualizer view("11");
    int v1;
    int v2;

    view.createViewPort(0, 0.0, 0.5, 1.0, v1);
    view.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    view.setBackgroundColor(0, 0, 0, v1);
    view.setBackgroundColor(0.05, 0, 0, v2);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> sources_cloud_color(source, 250, 0, 0);
    view.addPointCloud(source, sources_cloud_color, "sources_cloud_v1", v1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_cloud_color(cloud, 0, 250, 0);
    view.addPointCloud(cloud, target_cloud_color, "target_cloud_v1", v1);
    view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sources_cloud_v1");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligend_cloud_color(align, 255, 0, 0);
    view.addPointCloud(align, aligend_cloud_color, "aligend_cloud_v2", v2);
    view.addPointCloud(cloud, target_cloud_color, "target_cloud_v2", v2);
    view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "aligend_cloud_v2");
    view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud_v2");


    pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> crude_cor_est;
    boost::shared_ptr<pcl::Correspondences> cru_correspondences(new pcl::Correspondences);
    crude_cor_est.setInputSource(source_fpfh);
    crude_cor_est.setInputTarget(cloud_fpfh);
    //  crude_cor_est.determineCorrespondences(cru_correspondences);
    crude_cor_est.determineReciprocalCorrespondences(*cru_correspondences);
    cout << "crude size is:" << cru_correspondences->size() << endl;
    view.addCorrespondences<pcl::PointXYZ>(source, cloud, *cru_correspondences, "correspond", v1);//添加显示对应点对

    view.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 0.1,
                                     "correspond"); //设置对应点连线的粗细.PCL_VISUALIZER_LINE_WIDTH,表示线操作,线段的宽度为2（提醒一下自己: 线段的宽度最好不要超过自定义的点的大小）,"correspond"表示对 对应的标签 做处理.
    view.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1,
                                     "correspond"); //设置对应点连线的颜色，范围从0-1之间。

    //pcl::io::savePCDFile("crou_output.pcd", *align);

    cout << "代码段运行时间: " << time.toc() / 1000 << "s" << endl;//计算程序运行时间
    while (!view.wasStopped()) { // Display the visualiser until 'q' key is pressed
        view.spinOnce();
    }

    //system("pause");//不注释，命令行会提示按任意键继续，注释会直接跳出
    return 0;
}

