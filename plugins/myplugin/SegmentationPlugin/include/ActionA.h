// Example of a plugin action

#pragma once

#include <QDialog>
#include <QLabel>
#include <QLineEdit>
#include <QWidget>
#include <QPushButton>
#include <QObject>
#include <ccPointCloud.h>
#include <fstream>
#include <sstream>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <string>
#include <utility>
#include <pcl/surface/mls.h>
#include <vector>
#include <limits>


class ccMainAppInterface;

namespace Example
{
    void    performActionA(ccMainAppInterface* appInterface);
}

struct DisAttr {
    int indice_0;
    int indice_1;
    float dis;
};

class SegDialog : public QDialog {
    Q_OBJECT

public:
    explicit SegDialog(ccMainAppInterface* appInterface = nullptr, QWidget* parent = nullptr);

    ~SegDialog() {};

    void SetInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
        cloud_ = cloud;
    }

public:
    QLineEdit* min_edit_, * max_edit_, * angle_threshold_, * leaf_size_, * neibor_, * cur_threshold_;
    QLineEdit* tola_edit_, * min_eular_edit_;

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>
        SegDialog::RegionGrow(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_,
            float leaf_size = 0.01,
            int min_size = 500,
            int max_size = 100000,
            int neibor = 30,
            float angle_threshold = 2.0,
            float cur_threshold = 1.0);
private:
    ccMainAppInterface* appInterface_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_; // 输入点云
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters_; // 各簇点云
    float ComputeTrace(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    int count_;

    Eigen::Vector4d FindPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        pcl::PointIndices& inliers,
        double dist_threshold = 0.02,
        double probability = 0.99, int max_iterations = 50);

    std::vector<pcl::PointIndices> EuclideanCluster(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
        float tolerance = 0.02, int min_size = 100);

    std::vector<DisAttr> ComputeDis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointIndices indice, Eigen::Vector3f normal);
public slots:
    // combine QComboBox slot
    void execute_slot();
};
