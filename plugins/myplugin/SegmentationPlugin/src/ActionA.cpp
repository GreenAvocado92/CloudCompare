// Example of a plugin action

#include "ccMainAppInterface.h"
#include "ActionA.h"
#include <cmath>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

std::string name_;

namespace Example
{
    void performActionA(ccMainAppInterface* appInterface)
    {
        // 创建一个SegDialog类
        SegDialog* pDlg = new SegDialog(appInterface);
        pDlg->setMinimumSize(480, 640);
        pDlg->setWindowTitle("Region Grow Clusters");
        pDlg->setAttribute(Qt::WA_DeleteOnClose, false);
        if (appInterface == nullptr)
        {
            // The application interface should have already been initialized when the plugin is loaded
            Q_ASSERT(false);

            return;
        }

        // 获取选中的点云
        ccPointCloud* cloud;
        const ccHObject::Container& selectedEntities = appInterface->getSelectedEntities();
        if (selectedEntities.size() != 1)
        {
            appInterface->dispToConsole("[ExamplePlugin] selectedEntities.size() != 1", ccMainAppInterface::ERR_CONSOLE_MESSAGE);
            return;
        }

        ccHObject* entity = selectedEntities[0];
        if (entity->isA(CC_TYPES::POINT_CLOUD))
        {
            cloud = static_cast<ccPointCloud*>(entity);
        }
        else
        {
            appInterface->dispToConsole("[ExamplePlugin] is not a CC_TYPES::POINT_CLOUD", ccMainAppInterface::ERR_CONSOLE_MESSAGE);
            return;
        }
        QString str = " newPointCloud size = " + QString::number(cloud->size());
        appInterface->dispToConsole(str, ccMainAppInterface::STD_CONSOLE_MESSAGE);
        name_ = cloud->getName().toStdString();

        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (int i = 0; i < cloud->size(); i++) {
            const CCVector3* p = cloud->getPoint(static_cast<unsigned>(i));
            pcl_cloud->emplace_back(pcl::PointXYZ(static_cast<float>(p->x),
                static_cast<float>(p->y),
                static_cast<float>(p->z)));
        }
        // 设置点云
        pDlg->SetInputCloud(pcl_cloud);
        pDlg->show();
    }
}

#include <pcl/common/transforms.h>
void SegDialog::execute_slot() {
    // 解析参数
    float leaf_float = leaf_size_->text().toFloat();
    int min_number = min_edit_->text().toFloat();
    int max_number = max_edit_->text().toFloat();
    float angle_float = angle_threshold_->text().toFloat();
    float cur_float = cur_threshold_->text().toFloat();
    int neibor_number = neibor_->text().toFloat();

    float tola = tola_edit_->text().toFloat();
    int min_eular_num = min_eular_edit_->text().toInt();

    // 区域增长聚类
    clusters_ = RegionGrow(cloud_, leaf_float, min_number, max_number, neibor_number, angle_float, cur_float);

    // 在cc界面上展示效果
    appInterface_->dispToConsole("cluster number = " + QString::number(clusters_.size()), ccMainAppInterface::STD_CONSOLE_MESSAGE);

    ccHObject* ccClusters = new ccHObject();
    ccClusters->setName("clusters" + QString::number(count_));
    pcl::PointIndices inliers;

    // 存储每个簇的中心点
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_centers(new pcl::PointCloud<pcl::PointXYZ>);
    // 存储每个簇的法向量
    pcl::PointCloud<pcl::Normal>::Ptr cluster_centers_normal(new pcl::PointCloud<pcl::Normal>);

    std::vector<float> attributes(0); // 存储产状信息
    std::vector<Eigen::Vector3f> cluster_centroid(0);
    for (int i = 0; i < clusters_.size(); i++) {
        ccPointCloud* newPointCloud = new ccPointCloud();
        auto cluster = clusters_[i];
        float dis = ComputeTrace(cluster);
        // pcl::io::savePLYFile("./tmp/P" + std::to_string(i) + ".ply", *cluster);
        // 计算cluster的产状
        Eigen::Vector4d params = FindPlane(cluster, inliers); // 平面拟合
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero(); // 中心点计算
        for (const auto& it : cluster->points) {
            centroid(0) = centroid(0) + it.x;
            centroid(1) = centroid(1) + it.y;
            centroid(2) = centroid(2) + it.z;
        }
        centroid = centroid / cluster->size();
        cluster_centroid.emplace_back(centroid);
        pcl::PointXYZ xln;
        pcl::Normal nor;
        xln.x = centroid(0);
        xln.y = centroid(1);
        xln.z = centroid(2);
        cluster_centers->emplace_back(xln);
        nor.normal_x = params(0);
        nor.normal_y = params(1);
        nor.normal_z = params(2);
        cluster_centers_normal->emplace_back(nor);

        //平面法向
        Eigen::Vector3f ref_normal(0, 0, 1); // xy平面的法向
        Eigen::Vector3f cur_normal(params(0), params(1), params(2)); // 识别到的平面的法向
        if (cur_normal(2) < 0) cur_normal = -1 * cur_normal;
        std::string str_normal = "No." + std::to_string(i) + "  Nor:" + std::to_string(cur_normal(0)) + " " + std::to_string(cur_normal(1)) + " " + std::to_string(cur_normal(2));
        appInterface_->dispToConsole(QString::fromStdString(str_normal), ccMainAppInterface::STD_CONSOLE_MESSAGE);

        // 倾角计算
        float ang_offset = acos(params(2)) * 180 / 3.1415926;
        if (ang_offset < 0) ang_offset = acos(-1 * params(2)) * 180 / 3.1415926;
        ang_offset = 90 - abs(90 - ang_offset);

        // 倾向计算
        // referencr: https://en.wikipedia.org/wiki/Vector_projection
        Eigen::Vector3f proj = cur_normal - (cur_normal.dot(ref_normal)) * (ref_normal); // 计算投影向量

        float dir_offset = acos(proj(1)) * 180 / 3.1415926; // 计算与y轴（正东/西/南/北）的夹角
        if (proj(0) < 0) dir_offset = 360 - dir_offset;

        attributes.emplace_back(ang_offset);
        attributes.emplace_back(dir_offset);
        attributes.emplace_back(dis);
        for (auto it : cluster->points) {
            newPointCloud->addPoint(CCVector3(it.x, it.y, it.z));
        }
        char deg[] = { 0xa1,0xe3, 0 };
        newPointCloud->setName(QString("p") + QString::number(i)
            + QString("(Dip:") + QString::number(ang_offset)
            + QString(",Dip.dir:") + QString::number(dir_offset)
            + QString(",tra:") + QString::number(dis) + QString("\)"));

        newPointCloud->setColor(ccColor::Generator::Random());
        newPointCloud->showColors(true);
        newPointCloud->setPointSize(6);
        ccClusters->addChild(newPointCloud);
    }

    appInterface_->addToDB(ccClusters);
    appInterface_->refreshAll();

    // 保存文件
    std::ofstream out("Atest.txt"); // 可改
    out << "点云名：" << name_ << std::endl;

    // 计算间距
    std::vector<pcl::PointIndices> cluster_dis = EuclideanCluster(cluster_centers, tola, min_eular_num);
    out << "结构面分组数：" << cluster_dis.size() << std::endl;
    std::vector<std::vector<DisAttr>> disattr(0);

    // 第 it 组内间距计算
    int cluster_count = 1;
    for (const auto& it : cluster_dis) {
        // 计算中心点
        // pcl::PointCloud<pcl::PointXYZ>::Ptr dis_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        Eigen::Vector3f tmp_center = Eigen::Vector3f::Zero();
        for (int i = 0; i < it.indices.size(); i++) {
            tmp_center(0) = tmp_center(0) + cluster_centers->points[it.indices[i]].x;
            tmp_center(1) = tmp_center(1) + cluster_centers->points[it.indices[i]].y;
            tmp_center(2) = tmp_center(2) + cluster_centers->points[it.indices[i]].z;
            // dis_cloud->emplace_back(cluster_centers->points[it.indices[i]]);
        }
        tmp_center = tmp_center / it.indices.size();

        // 选择距离中心点最近的点
        float min_dis = std::numeric_limits<float>::max();
        int min_dis_index = 0;
        for (int i = 0; i < it.indices.size(); i++) {
            float dis = (cluster_centers->points[it.indices[i]].x - tmp_center(0)) * (cluster_centers->points[it.indices[i]].x - tmp_center(0)) +
                (cluster_centers->points[it.indices[i]].y - tmp_center(1)) * (cluster_centers->points[it.indices[i]].y - tmp_center(1)) +
                (cluster_centers->points[it.indices[i]].z - tmp_center(2)) * (cluster_centers->points[it.indices[i]].z - tmp_center(2));
            if (min_dis > dis) {
                min_dis = dis;
                min_dis_index = i;
            }
        }

        out << "优势组" + std::to_string(cluster_count) + " 结构面数量共 " + std::to_string(it.indices.size()) + " 条" << std::endl;

        // 提取法向
        Eigen::Vector3f tmp_normal(cluster_centers_normal->points[min_dis_index].normal_x,
            cluster_centers_normal->points[min_dis_index].normal_y,
            cluster_centers_normal->points[min_dis_index].normal_z);

        // 计算间距
        auto dis_clu = ComputeDis(cluster_centers, it, tmp_normal);
        disattr.emplace_back(dis_clu);

        // 输出聚类中心的法向量，产状及间距
        std::string str = "";
        str = "Normal:" + std::to_string(tmp_normal(0)) + " " +
            std::to_string(tmp_normal(1)) + " " +
            std::to_string(tmp_normal(2));
        str = str + " Dip: " + std::to_string(attributes[min_dis_index * 3]) +
            " Dip.dir: " + std::to_string(attributes[min_dis_index * 3 + 1]) + " Dis:";
        // for (auto dis_tmp : dis_clu) str = str + std::to_string(dis_tmp.dis) + ",";

        out << str << std::endl;

        appInterface_->dispToConsole(QString::fromStdString(str), ccMainAppInterface::STD_CONSOLE_MESSAGE);
        cluster_count++;
    }

    count_++;

    for (int i = 0; i < cluster_dis.size(); i++) {
        out << "优势组" << std::to_string(i + 1) << std::endl;
        auto each_cluster = cluster_dis[i];
        for (int j = 0; j < each_cluster.indices.size(); j++) {
            out << "P" << std::to_string(each_cluster.indices[j]) << "; normal: " <<
                std::to_string(cluster_centers_normal->points[each_cluster.indices[j]].normal_x) << " " <<
                std::to_string(cluster_centers_normal->points[each_cluster.indices[j]].normal_y) << " " <<
                std::to_string(cluster_centers_normal->points[each_cluster.indices[j]].normal_z) << " " <<
                ";dip: " << std::to_string(attributes[each_cluster.indices[j] * 3]) << " ;dip.dir: " <<
                std::to_string(attributes[each_cluster.indices[j] * 3 + 1]) <<
                " ;tra: " << std::to_string(attributes[each_cluster.indices[j] * 3 + 2]) <<
                " ;centroid: " << cluster_centroid[each_cluster.indices[j]].transpose() << std::endl;
        }

        auto jianju = disattr[i];
        for (auto it : jianju) {
            out << "Dis between P" << std::to_string(it.indice_0) <<
                " and P" << std::to_string(it.indice_1) << ": " << std::to_string(it.dis) << std::endl;
        }

    }
    out.close();
}

std::vector<DisAttr> SegDialog::ComputeDis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointIndices indice, Eigen::Vector3f normal) {
    std::vector<DisAttr> dis(0);
    //  // 计算投影向量
    for (int j = 0; j < indice.indices.size() - 1; j++) {
        int i = indice.indices[j];
        Eigen::Vector3f v(cloud->points[i].x - cloud->points[i + 1].x,
            cloud->points[i].y - cloud->points[i + 1].y,
            cloud->points[i].z - cloud->points[i + 1].z);
        Eigen::Vector3f proj = v - (v.dot(normal)) * (normal);
        DisAttr  d;
        d.indice_0 = i;
        d.indice_1 = i + 1;
        d.dis = proj.norm();
        dis.emplace_back(d);
    }
    return dis;
}

// 构造函数
SegDialog::SegDialog(ccMainAppInterface* appInterface, QWidget* parent)
    : appInterface_(appInterface), QDialog(parent) {
    cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    clusters_.resize(0);
    count_ = 0;
    QLabel* leaf_size = new QLabel("LeafSize", this);
    leaf_size->setGeometry(20, 20, 150, 30);
    leaf_size_ = new QLineEdit("0.01", this);
    leaf_size_->setGeometry(180, 20, 100, 30);

    QLabel* min_size = new QLabel("MinNum", this);
    min_size->setGeometry(20, 60, 150, 30);
    min_edit_ = new QLineEdit("500", this);
    min_edit_->setGeometry(180, 60, 100, 30);

    QLabel* max_size = new QLabel("MaxNum", this);
    max_size->setGeometry(20, 100, 150, 30);
    max_edit_ = new QLineEdit("1000000", this);
    max_edit_->setGeometry(180, 100, 100, 30);

    QLabel* angle_threshold = new QLabel("Angle", this);
    angle_threshold->setGeometry(20, 140, 150, 30);
    angle_threshold_ = new QLineEdit("2", this);
    angle_threshold_->setGeometry(180, 140, 100, 30);

    QLabel* cur_threshold = new QLabel("Curver", this);
    cur_threshold->setGeometry(20, 180, 150, 30);
    cur_threshold_ = new QLineEdit("1", this);
    cur_threshold_->setGeometry(180, 180, 100, 30);

    QLabel* neibor = new QLabel("Neighbor", this);
    neibor->setGeometry(20, 220, 150, 30);
    neibor_ = new QLineEdit("30", this);
    neibor_->setGeometry(180, 220, 100, 30);


    QLabel* tola = new QLabel("Tolerance", this);
    tola->setGeometry(20, 260, 150, 30);
    tola_edit_ = new QLineEdit("0.5", this);
    tola_edit_->setGeometry(180, 260, 100, 30);

    QLabel* min_eular = new QLabel("MinCluNum", this);
    min_eular->setGeometry(20, 300, 150, 30);
    min_eular_edit_ = new QLineEdit("4", this);
    min_eular_edit_->setGeometry(180, 300, 100, 30);

    QPushButton* execute = new QPushButton("Exe", this);
    execute->setGeometry(10, 350, 150, 30);
    // 槽函数
    QObject::connect(execute, SIGNAL(clicked()), this, SLOT(execute_slot()));

}

// RANSAC算法进行平面拟合
Eigen::Vector4d SegDialog::FindPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    pcl::PointIndices& inliers,
    double dist_threshold,
    double probability, int max_iterations) {
    pcl::ModelCoefficients coefficients;
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg(true);
    // Optional
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(dist_threshold);
    seg.setProbability(probability);
    seg.setMaxIterations(max_iterations);
    seg.setInputCloud(cloud);
    // 分割出平面点云
    seg.segment(inliers, coefficients);

    // 将平面参数改成Eigen::Vector4d格式
    Eigen::Vector4d plane_coefficients;
    for (size_t i = 0; i < coefficients.values.size(); ++i) {
        plane_coefficients(i) = coefficients.values[i];
    }
    return plane_coefficients;
}

// 区域生长聚类算法
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>
SegDialog::RegionGrow(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_,
    float leaf_size,
    int min_size,
    int max_size,
    int neibor,
    float angle_threshold,
    float cur_threshold)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // vg是降采样类
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.setInputCloud(cloud_);
    vg.filter(*cloud);

    // 法向量估计
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(18);
    ne.compute(*normals);

    // 区域增长聚类
    // pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(min_size);
    reg.setMaxClusterSize(max_size);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(neibor); // 30
    reg.setInputCloud(cloud);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(angle_threshold / 180.0 * M_PI); // 角度阈值
    reg.setCurvatureThreshold(cur_threshold);
    std::vector<pcl::PointIndices> indices(0);
    reg.extract(indices);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters(0);
    for (const auto& it : indices) {
        // 区域增长分割的输出结果是样点的索引，下面就是将索引转成样点
        pcl::PointIndices::Ptr target_indices(new pcl::PointIndices(it));
        pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_cloud(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extractor(false);
        extractor.setInputCloud(cloud);
        extractor.setIndices(target_indices);
        extractor.setNegative(false);
        extractor.filter(*segmented_cloud);
        clusters.emplace_back(segmented_cloud);
    }
    return clusters;
}

// 欧式聚类分割
std::vector<pcl::PointIndices> SegDialog::EuclideanCluster(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    float tolerance, int min_size) {
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> cluster_extractor;

    cluster_extractor.setClusterTolerance(tolerance);
    cluster_extractor.setMinClusterSize(min_size);
    cluster_extractor.setInputCloud(cloud);
    cluster_extractor.extract(cluster_indices);

    return cluster_indices;
}

// 计算迹长：最大距离
float SegDialog::ComputeTrace(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    float dis = 0.0;
    if (cloud->size() < 2) return dis;
    for (int i = 0; i < cloud->size() - 1; i++) {
        auto st = cloud->points[i];
        for (int j = i + 1; j < cloud->size(); j++) {
            auto en = cloud->points[j];
            float dis_tmp = (en.x - st.x) * (en.x - st.x) +
                (en.y - st.y) * (en.y - st.y) +
                (en.z - st.z) * (en.z - st.z);
            if (dis < dis_tmp) dis = dis_tmp;
        }
    }
    return sqrt(dis);
}
