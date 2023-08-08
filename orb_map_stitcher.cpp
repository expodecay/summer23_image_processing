#include "mpi.h"
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;

const int MAXBYTES = 100000000; // 8 * 1024 * 1024;
uchar* buffer = nullptr;

struct image_info {
    int width;
    int height;
    int type;
    int channels;
};

static void printUsage(char** argv)
{
    cout <<
        "Rotation model images stitcher.\n\n"
        << argv[0] << " img1 img2 [...imgN] [flags]\n\n"
        "Flags:\n"
        "  --preview\n"
        "      Run stitching in the preview mode. Works faster than usual mode,\n"
        "      but output image will have lower resolution.\n"
        "  --try_cuda (yes|no)\n"
        "      Try to use CUDA. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "\nMotion Estimation Flags:\n"
        "  --work_megapix <float>\n"
        "      Resolution for image registration step. The default is 0.6 Mpx.\n"
        "  --features (surf|orb|sift|akaze)\n"
        "      Type of features used for images matching.\n"
        "      The default is surf if available, orb otherwise.\n"
        "  --matcher (homography|affine)\n"
        "      Matcher used for pairwise image matching.\n"
        "  --estimator (homography|affine)\n"
        "      Type of estimator used for transformation estimation.\n"
        "  --match_conf <float>\n"
        "      Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
        "  --conf_thresh <float>\n"
        "      Threshold for two images are from the same panorama confidence.\n"
        "      The default is 1.0.\n"
        "  --ba (no|reproj|ray|affine)\n"
        "      Bundle adjustment cost function. The default is ray.\n"
        "  --ba_refine_mask (mask)\n"
        "      Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
        "      where 'x' means refine respective parameter and '_' means don't\n"
        "      refine one, and has the following format:\n"
        "      <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
        "      adjustment doesn't support estimation of selected parameter then\n"
        "      the respective flag is ignored.\n"
        "  --wave_correct (no|horiz|vert)\n"
        "      Perform wave effect correction. The default is 'horiz'.\n"
        "  --save_graph <file_name>\n"
        "      Save matches graph represented in DOT language to <file_name> file.\n"
        "      Labels description: Nm is number of matches, Ni is number of inliers,\n"
        "      C is confidence.\n"
        "\nCompositing Flags:\n"
        "  --warp (affine|plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
        "      Warp surface type. The default is 'spherical'.\n"
        "  --seam_megapix <float>\n"
        "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
        "  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
        "      Seam estimation method. The default is 'gc_color'.\n"
        "  --compose_megapix <float>\n"
        "      Resolution for compositing step. Use -1 for original resolution.\n"
        "      The default is -1.\n"
        "  --expos_comp (no|gain|gain_blocks|channels|channels_blocks)\n"
        "      Exposure compensation method. The default is 'gain_blocks'.\n"
        "  --expos_comp_nr_feeds <int>\n"
        "      Number of exposure compensation feed. The default is 1.\n"
        "  --expos_comp_nr_filtering <int>\n"
        "      Number of filtering iterations of the exposure compensation gains.\n"
        "      Only used when using a block exposure compensation method.\n"
        "      The default is 2.\n"
        "  --expos_comp_block_size <int>\n"
        "      BLock size in pixels used by the exposure compensator.\n"
        "      Only used when using a block exposure compensation method.\n"
        "      The default is 32.\n"
        "  --blend (no|feather|multiband)\n"
        "      Blending method. The default is 'multiband'.\n"
        "  --blend_strength <float>\n"
        "      Blending strength from [0,100] range. The default is 5.\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n"
        "  --timelapse (as_is|crop) \n"
        "      Output warped images separately as frames of a time lapse movie, with 'fixed_' prepended to input file names.\n"
        "  --rangewidth <int>\n"
        "      uses range_width to limit number of images to match with.\n";
}


// Default command line args
vector<String> img_names_input;
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 0.3f;
string features_type = "orb";
float match_conf = 0.3f;
string matcher_type = "affine";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "x____";
bool do_wave_correct = false;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::NO;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;
string seam_find_type = "gc_color";
int blend_type = Blender::NO;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = 1;

static int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage(argv);
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage(argv);
            return -1;
        }
        else if (string(argv[i]) == "--preview")
        {
            preview = true;
        }
        else if (string(argv[i]) == "--try_cuda")
        {
            if (string(argv[i + 1]) == "no")
                try_cuda = false;
            else if (string(argv[i + 1]) == "yes")
                try_cuda = true;
            else
            {
                cout << "Bad --try_cuda flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--work_megapix")
        {
            work_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--seam_megapix")
        {
            seam_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--compose_megapix")
        {
            compose_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--result")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--features")
        {
            features_type = argv[i + 1];
            if (string(features_type) == "orb")
                match_conf = 0.3f;
            i++;
        }
        else if (string(argv[i]) == "--matcher")
        {
            if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
                matcher_type = argv[i + 1];
            else
            {
                cout << "Bad --matcher flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--estimator")
        {
            if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
                estimator_type = argv[i + 1];
            else
            {
                cout << "Bad --estimator flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--match_conf")
        {
            match_conf = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--conf_thresh")
        {
            conf_thresh = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--ba")
        {
            ba_cost_func = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--ba_refine_mask")
        {
            ba_refine_mask = argv[i + 1];
            if (ba_refine_mask.size() != 5)
            {
                cout << "Incorrect refinement mask length.\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--wave_correct")
        {
            if (string(argv[i + 1]) == "no")
                do_wave_correct = false;
            else if (string(argv[i + 1]) == "horiz")
            {
                do_wave_correct = true;
                wave_correct = detail::WAVE_CORRECT_HORIZ;
            }
            else if (string(argv[i + 1]) == "vert")
            {
                do_wave_correct = true;
                wave_correct = detail::WAVE_CORRECT_VERT;
            }
            else
            {
                cout << "Bad --wave_correct flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--save_graph")
        {
            save_graph = true;
            save_graph_to = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--warp")
        {
            warp_type = string(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp")
        {
            if (string(argv[i + 1]) == "no")
                expos_comp_type = ExposureCompensator::NO;
            else if (string(argv[i + 1]) == "gain")
                expos_comp_type = ExposureCompensator::GAIN;
            else if (string(argv[i + 1]) == "gain_blocks")
                expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
            else if (string(argv[i + 1]) == "channels")
                expos_comp_type = ExposureCompensator::CHANNELS;
            else if (string(argv[i + 1]) == "channels_blocks")
                expos_comp_type = ExposureCompensator::CHANNELS_BLOCKS;
            else
            {
                cout << "Bad exposure compensation method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--expos_comp_nr_feeds")
        {
            expos_comp_nr_feeds = atoi(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp_nr_filtering")
        {
            expos_comp_nr_filtering = atoi(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp_block_size")
        {
            expos_comp_block_size = atoi(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--seam")
        {
            if (string(argv[i + 1]) == "no" ||
                string(argv[i + 1]) == "voronoi" ||
                string(argv[i + 1]) == "gc_color" ||
                string(argv[i + 1]) == "gc_colorgrad" ||
                string(argv[i + 1]) == "dp_color" ||
                string(argv[i + 1]) == "dp_colorgrad")
                seam_find_type = argv[i + 1];
            else
            {
                cout << "Bad seam finding method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--blend")
        {
            if (string(argv[i + 1]) == "no")
                blend_type = Blender::NO;
            else if (string(argv[i + 1]) == "feather")
                blend_type = Blender::FEATHER;
            else if (string(argv[i + 1]) == "multiband")
                blend_type = Blender::MULTI_BAND;
            else
            {
                cout << "Bad blending method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--timelapse")
        {
            timelapse = true;

            if (string(argv[i + 1]) == "as_is")
                timelapse_type = Timelapser::AS_IS;
            else if (string(argv[i + 1]) == "crop")
                timelapse_type = Timelapser::CROP;
            else
            {
                cout << "Bad timelapse method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--rangewidth")
        {
            range_width = atoi(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--blend_strength")
        {
            blend_strength = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else
            img_names_input.push_back(argv[i]);
    }
    if (preview)
    {
        compose_megapix = 0.6;
    }
    return 0;
}

Mat image_stitch(Mat img_0, Mat img_1, int rank, string image_two) {
    int num_images = 2;
    vector<Mat> imagesList;
    imagesList.push_back(img_0);
    imagesList.push_back(img_1);
    Mat Stitch_result;

    cout << "============= processor[" << rank << "]============ loading image... " << image_two << "   size 0: " << imagesList[0].size() <<  " size 1: " << imagesList[1].size() << endl;

    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    LOGLN("Finding features...");
#if ENABLE_LOG
    int64 t = getTickCount();
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
#endif

    Ptr<Feature2D> finder;
    if (features_type == "orb")
    {
        finder = ORB::create(5000, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 0);
    }
    else if (features_type == "akaze")
    {
        finder = AKAZE::create();
    }
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (features_type == "surf")
    {
        finder = xfeatures2d::SURF::create();
    }
#endif
    else if (features_type == "sift")
    {
        finder = SIFT::create();
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return Stitch_result;
    }

    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    for (int i = 0; i < num_images; ++i)
    {
        //full_img = imread(samples::findFile(img_names[i]));
        full_img = imagesList[i];
        full_img_sizes[i] = full_img.size();

        if (full_img.empty())
        {
            LOGLN("Can't open image " << imagesList[i]);
            return Stitch_result;
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        computeImageFeatures(finder, img, features[i]);
        features[i].img_idx = i;
        LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());

        resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
        images[i] = img.clone();
    }

    full_img.release();
    img.release();

    //LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Finding features, time: " << diff.count() << " s\n";

    LOG("Pairwise matching");
#if ENABLE_LOG
    t = getTickCount();
    start = std::chrono::high_resolution_clock::now();
#endif
    vector<MatchesInfo> pairwise_matches;
    Ptr<FeaturesMatcher> matcher;
    if (matcher_type == "affine")
    {
        cout << "######Try_CUDA: matcher_type == affine\n";
        cout << "makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf)\n";
        matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);

    }
    else if (range_width == -1)
    {
        cout << "######Try_CUDA:range_width==-1\n";
        cout << "makePtr<BestOf2NearestMatcher>(try_cuda, match_conf)\n";
        matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
    }
    else
    {
        cout << "######Try_CUDA\n";
        cout << "makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf)\n";
        matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);
    }

    (*matcher)(features, pairwise_matches);
    matcher->collectGarbage();

    // Estimate rotation using good matches
    bool rotated = false;
    for (size_t i = 0; i < pairwise_matches.size(); ++i) {
        const MatchesInfo& match_info = pairwise_matches[i];
        if (match_info.num_inliers > 5) {
            std::cout << match_info.num_inliers << "\n";
            Mat H = match_info.H;
            double rotation_rad = atan2(H.at<double>(1, 0), H.at<double>(0, 0));
            double rotation_deg = rotation_rad * 180.0 / CV_PI;
            if (abs(rotation_deg) > 10) rotated = true;
            cout << "Estimated Rotation Angle for image pair " << i << ": " << rotation_deg << " degrees" << endl;
        }
        else {
            std::cout << match_info.num_inliers << "\n";

            
            cout << "Not enough inliers for rotation estimation in image pair " << i << "." << endl;
        }
    }

    LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Pairwise matching, time: " << diff.count() << " s\n";


    // Check if we should save matches graph
    /*
    if (save_graph)
    {
        LOGLN("Saving matches graph...");
        ofstream f(save_graph_to.c_str());
        f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
    }

    // Leave only images we are sure are from the same panorama
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<String> img_names_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;

    // Check if we still have enough images
    num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }
    */

    Ptr<Estimator> estimator;
    if (estimator_type == "affine")
        estimator = makePtr<AffineBasedEstimator>();
    else
        estimator = makePtr<HomographyBasedEstimator>();

    vector<CameraParams> cameras;
    if (!(*estimator)(features, pairwise_matches, cameras))
    {
        cout << "Homography estimation failed.\n";
        return Stitch_result;
    }

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        //LOGLN("Initial camera intrinsics #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
    }

    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
    else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
    else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
    else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
    else
    {
        cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
        return Stitch_result;
    }
    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
    adjuster->setRefinementMask(refine_mask);
    if (!(*adjuster)(features, pairwise_matches, cameras))
    {
        cout << "Camera parameters adjusting failed.\n";
        return Stitch_result;
    }

    // Find median focal length

    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        //LOGLN("Camera #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;


    if (do_wave_correct)
    {
        std::cout << "=============YES==============\n";
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }
    else std::cout << "=============NO==============\n";

    LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
    t = getTickCount();
    start = std::chrono::high_resolution_clock::now();
#endif

    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);

    // Prepare images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks

    Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
    if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane")
        {
            cout << "######Try_cuda: warp_type == plane;\n";
            warper_creator = makePtr<cv::PlaneWarperGpu>();
        }
        else if (warp_type == "cylindrical")
        {
            cout << "######Try_cuda: warp_type == cylindrical;\n";
            warper_creator = makePtr<cv::CylindricalWarperGpu>();
        }
        else if (warp_type == "spherical")
        {
            cout << "######Try_cuda: warp_type == spherical;\n";
            warper_creator = makePtr<cv::SphericalWarperGpu>();
        }

    }
    else
#endif
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarper>();
        else if (warp_type == "affine")
            warper_creator = makePtr<cv::AffineWarper>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarper>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarper>();
        else if (warp_type == "fisheye")
            warper_creator = makePtr<cv::FisheyeWarper>();
        else if (warp_type == "stereographic")
            warper_creator = makePtr<cv::StereographicWarper>();
        else if (warp_type == "compressedPlaneA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlaneA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniA2B1")
            warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniA1.5B1")
            warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniPortraitA2B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniPortraitA1.5B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "mercator")
            warper_creator = makePtr<cv::MercatorWarper>();
        else if (warp_type == "transverseMercator")
            warper_creator = makePtr<cv::TransverseMercatorWarper>();
    }

    if (!warper_creator)
    {
        cout << "Can't create the following warper '" << warp_type << "'\n";
        return Stitch_result;
    }

    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0, 0) *= swa; K(0, 2) *= swa;
        K(1, 1) *= swa; K(1, 2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    //LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Warping images, time: " << diff.count() << " s\n";

    LOGLN("Compensating exposure...");
#if ENABLE_LOG
    t = getTickCount();
    start = std::chrono::high_resolution_clock::now();
#endif

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    if (dynamic_cast<GainCompensator*>(compensator.get()))
    {
        GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
        gcompensator->setNrFeeds(expos_comp_nr_feeds);
    }

    if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
    {
        ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
        ccompensator->setNrFeeds(expos_comp_nr_feeds);
    }

    if (dynamic_cast<BlocksCompensator*>(compensator.get()))
    {
        BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
        bcompensator->setNrFeeds(expos_comp_nr_feeds);
        bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
        bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
    }

    compensator->feed(corners, images_warped, masks_warped);

    //LOGLN("Compensating exposure, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Compensating exposure, time: " << diff.count() << " s\n";

    LOGLN("Finding seams...");
#if ENABLE_LOG
    t = getTickCount();
    start = std::chrono::high_resolution_clock::now();
#endif

    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = makePtr<detail::NoSeamFinder>();
    else if (seam_find_type == "voronoi")
        seam_finder = makePtr<detail::VoronoiSeamFinder>();
    else if (seam_find_type == "gc_color")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        ////////
        /*
            if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            {
                cout << "Try_cuda: seam_find_type == gc_color;\n";
                seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
            }

            else*/
#endif
        seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
        {
            cout << "######Try_cuda: seam_find_type == gc_colorgrad\n";
            seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        }

        else
#endif
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
    if (!seam_finder)
    {
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return Stitch_result;
    }

    seam_finder->find(images_warped_f, corners, masks_warped);

    //LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Finding seams, time: " << diff.count() << " s\n";

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    LOGLN("Compositing...");
#if ENABLE_LOG
    t = getTickCount();
    start = std::chrono::high_resolution_clock::now();
#endif

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    Ptr<Timelapser> timelapser;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        //LOGLN("Compositing image #" << indices[img_idx]+1);

        // Read image and resize it if necessary
        //full_img = imread(samples::findFile(img_names[img_idx]));
        full_img = imagesList[img_idx];

        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // Compensate exposure
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
        mask_warped = seam_mask & mask_warped;

        if (!blender && !timelapse)
        {
            cout << "######Try_CUDA\n";
            cout << "blender = Blender::createDefault(blend_type, try_cuda);\n";
            blender = Blender::createDefault(blend_type, try_cuda);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
            {
                cout << "######Try_CUDA: blend_width < 1.f\n";
                cout << "blender = Blender::createDefault(Blender::NO, try_cuda); \n";
                blender = Blender::createDefault(Blender::NO, try_cuda);
            }

            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
                LOGLN("Multi-band blender, number of bands: " << mb->numBands());
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
                fb->setSharpness(1.f / blend_width);
                LOGLN("Feather blender, sharpness: " << fb->sharpness());
            }
            blender->prepare(corners, sizes);
        }
        else if (!timelapser && timelapse)
        {
            timelapser = Timelapser::createDefault(timelapse_type);
            timelapser->initialize(corners, sizes);
        }

        // Blend the current image
        if (timelapse)
        {
            /*
            timelapser->process(img_warped_s, Mat::ones(img_warped_s.size(), CV_8UC1), corners[img_idx]);
            String fixedFileName;
            size_t pos_s = String(img_names[img_idx]).find_last_of("/\\");
            if (pos_s == String::npos)
            {
                fixedFileName = "fixed_" + img_names[img_idx];
            }
            else
            {
                fixedFileName = "fixed_" + String(img_names[img_idx]).substr(pos_s + 1, String(img_names[img_idx]).length() - pos_s);
            }
            imwrite(fixedFileName, timelapser->getDst());
            */
        }
        else
        {
            blender->feed(img_warped_s, mask_warped, corners[img_idx]);
        }
    }

    if (!timelapse)
    {
        Mat result, result_mask;
        blender->blend(result, result_mask);

        //LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
        //imwrite(result_name, result);
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        std::cout << "Compositing, time: " << diff.count() << " s\n";
        Stitch_result = result;
    }
    Stitch_result.convertTo(Stitch_result, CV_8UC1);

    return Stitch_result;

}

void matsnd(const Mat& m, int dest, int rank) {

    int rows = m.rows;
    int cols = m.cols;
    int type = m.type();
    int channels = m.channels();

    // instance of image info
    image_info a = { rows, cols, type, channels };
    //memcpy(buffer, &a, sizeof(a));  // also works
    memcpy(buffer, &a, sizeof(image_info)); // image size

    memcpy(buffer + sizeof(image_info), m.data, sizeof(uchar) * rows * cols * 3); // actual data (with offset)

    /*memcpy(&buffer[0 * sizeof(int)], (uchar*)&rows, sizeof(int));
    memcpy(&buffer[1 * sizeof(int)], (uchar*)&cols, sizeof(int));
    memcpy(&buffer[2 * sizeof(int)], (uchar*)&type, sizeof(int));*/

    int bytespersample = 1; // change if using shorts or floats
    int bytes = (m.rows * m.cols * channels * bytespersample) + sizeof(image_info);

    cout << "Process " << rank << ": sending results..." << endl;

    cout << "matsnd: rows=" << rows << endl;
    cout << "matsnd: cols=" << cols << endl;
    cout << "matsnd: type=" << type << endl;
    cout << "matsnd: channels=" << channels << endl;
    cout << "matsnd: bytes=" << bytes << "\n" << endl;

    if (!m.isContinuous())
    {
        m == m.clone();
    }

    /*std::ofstream file;
    file.open("fileNameSnd");
    file.write((const char*)buffer, bytes);
    file.close();*/

    //memcpy(&buffer[3 * sizeof(int)], m.data, bytes);
    MPI_Send(buffer, bytes, MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
}

Mat matrcv(int src, uchar* buf, int rank) {

    MPI_Status status;
    int count, rows, cols, type, channels;

    //MPI_Recv(&buffer, sizeof(buffer), MPI_UNSIGNED_CHAR, src, 0, MPI_COMM_WORLD, &status);
    //MPI_Recv(buffer, sizeof(uchar)*3873*2929*3, MPI_UNSIGNED_CHAR, src, 0, MPI_COMM_WORLD, &status);

    //int err = MPI_Recv(buf, buf_size, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    //MPI_Recv(buf, buf_size, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(buf, MAXBYTES, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    image_info* info = reinterpret_cast<image_info*>(buf); // mapping info to the portion of buf that points to image_info
    uchar* image_buf = buf + sizeof(image_info); // ofset the pointer to ignore image_info and start where actual data starts

    int image_size = (info->width * info->height * info->channels); // size of the image data

    /*std::ofstream file;
    file.open("fileName");
    file.write((const char*)buffer, image_size);
    file.close();*/

    MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);

    /*memcpy((uchar*)&rows, &buffer[0 * sizeof(int)], sizeof(int));
    memcpy((uchar*)&cols, &buffer[1 * sizeof(int)], sizeof(int));
    memcpy((uchar*)&type, &buffer[2 * sizeof(int)], sizeof(int));*/

    cout << "Process " << rank << ": receiving results..." << endl;

    /*rows = 3873;
    cols = 2929;
    type = 16;*/
    rows = info->width; // -> is the dereference operator (similar to .)
    cols = info->height;
    type = info->type;

    cout << "matrcv: Count=" << count << endl;
    cout << "matrcv: rows=" << rows << endl;
    cout << "matrcv: cols=" << cols << endl;
    cout << "matrcv: type=" << type << endl;

    // Make the mat

    Mat received = Mat(rows, cols, type, image_buf);

    received.convertTo(received, CV_8UC1);

    /*imshow("Image", received);
    waitKey(1);*/

    return received;
}

int main(int argc, char** argv) {

#if ENABLE_LOG
    auto start = std::chrono::high_resolution_clock::now();
#endif

#if 0
    cv::setBreakOnError(true);
#endif

    buffer = new uchar[MAXBYTES];
    int x = 3;

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /*  int temp = 0;
      if (world_rank == 0) {
          cout << "press any key to continue..." << endl;
          cin >> temp;
      }*/



    MPI_Barrier(MPI_COMM_WORLD);

    int retval = parseCmdArgs(argc, argv);

    int num_images_input = static_cast<int>(img_names_input.size());

    Mat result, img_0, img_1, partial_result;




    /********************************************************************************
     *                                                                              *
     *                                  N=8                                         *
     *                                                                              *
    ********************************************************************************/
    //if (world_rank == 0) {

    //    LOGLN("Loading....................." << img_names_input[0]);
    //    img_0 = imread(samples::findFile(img_names_input[0]));

    //    LOGLN("Loading....................." << img_names_input[1]);
    //    img_1 = imread(samples::findFile(img_names_input[1]));

    //    // I_0_1
    //    partial_result = image_stitch(img_0, img_1);
    //    partial_result.convertTo(partial_result, CV_8UC1);

    //    img_0.release();
    //    img_1.release();

    //    //I_0_1_2_3
    //    img_0 = partial_result;
    //    img_1 = matrcv(4, buffer, world_rank);
    //    partial_result = image_stitch(img_0, img_1);
    //            
    //    //I_0_1_2_3_4_5_6_7
    //    img_0 = partial_result;
    //    img_1 = matrcv(1, buffer, world_rank);
    //    partial_result = image_stitch(img_0, img_1);

    //    //I_0_1_2_3_4_5_6_7_8_9_10_11_12_14_15_16
    //    img_0 = partial_result;
    //    img_1 = matrcv(2, buffer, world_rank);
    //    partial_result = image_stitch(img_0, img_1);
    //    imwrite(format("process_%d.png", world_rank), partial_result);

    //    partial_result.release();
    //}

    //if (world_rank == 1) {

    //    LOGLN("Loading....................." << img_names_input[4]);
    //    img_0 = imread(samples::findFile(img_names_input[4]));

    //    LOGLN("Loading....................." << img_names_input[5]);
    //    img_1 = imread(samples::findFile(img_names_input[5]));

    //    //I_4_5
    //    partial_result = image_stitch(img_0, img_1);
    //            
    //    img_0.release();
    //    img_1.release();

    //    //I_4_5_6_7
    //    img_0 = partial_result;
    //    img_1 = matrcv(5, buffer, world_rank);
    //    partial_result = image_stitch(img_0, img_1);

    //    matsnd(partial_result, 0, world_rank);
    //    partial_result.release();
    //}

    //if (world_rank == 2) {

    //    LOGLN("Loading....................." << img_names_input[8]);
    //    img_0 = imread(samples::findFile(img_names_input[8]));

    //    LOGLN("Loading....................." << img_names_input[9]); //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    img_1 = imread(samples::findFile(img_names_input[9]));

    //    //I_8_9
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0.release();
    //    img_1.release();

    //    //I_8_9_10_11
    //    img_0 = partial_result;
    //    img_1 = matrcv(6, buffer, world_rank);
    //    partial_result = image_stitch(img_0, img_1);

    //    //I_8_9_10_11_12_14_15_16
    //    img_0 = partial_result;
    //    img_1 = matrcv(3, buffer, world_rank);
    //    partial_result = image_stitch(img_0, img_1);

    //    matsnd(partial_result, 0, world_rank);
    //    partial_result.release();
    //}

    //if (world_rank == 3) {

    //    LOGLN("Loading....................." << img_names_input[12]);
    //    img_0 = imread(samples::findFile(img_names_input[12]));

    //    LOGLN("Loading....................." << img_names_input[14]);
    //    img_1 = imread(samples::findFile(img_names_input[14]));

    //    //I_12_14
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0.release();
    //    img_1.release();

    //    //I_12_14_15_16
    //    img_0 = partial_result;
    //    img_1 = matrcv(7, buffer, world_rank);
    //    partial_result = image_stitch(img_0, img_1);

    //    matsnd(partial_result, 2, world_rank);
    //    partial_result.release();
    //}
    //        
    //if (world_rank == 4) {

    //    LOGLN("Loading....................." << img_names_input[2]);
    //    img_0 = imread(samples::findFile(img_names_input[2]));

    //    LOGLN("Loading....................." << img_names_input[3]);
    //    img_1 = imread(samples::findFile(img_names_input[3]));

    //    //I_2_3
    //    partial_result = image_stitch(img_0, img_1);
    //           
    //    img_0.release();
    //    img_1.release();

    //    matsnd(partial_result, 0, world_rank);

    //    partial_result.release();
    //}

    //if (world_rank == 5) {

    //    LOGLN("Loading....................." << img_names_input[6]);
    //    img_0 = imread(samples::findFile(img_names_input[6]));

    //    LOGLN("Loading....................." << img_names_input[7]); //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    img_1 = imread(samples::findFile(img_names_input[7]));

    //    //I_6_7
    //    partial_result = image_stitch(img_0, img_1);
    //           
    //    img_0.release();
    //    img_1.release();

    //    matsnd(partial_result, 1, world_rank);

    //    partial_result.release();
    //}

    //if (world_rank == 6) {

    //    LOGLN("Loading....................." << img_names_input[10]);
    //    img_0 = imread(samples::findFile(img_names_input[10]));

    //    LOGLN("Loading....................." << img_names_input[11]);
    //    img_1 = imread(samples::findFile(img_names_input[11]));

    //    //I_10_11
    //    partial_result = image_stitch(img_0, img_1);
    //            
    //    img_0.release();
    //    img_1.release();

    //    matsnd(partial_result, 2, world_rank);

    //    partial_result.release();
    //}

    //if (world_rank == 7) {

    //    LOGLN("Loading....................." << img_names_input[14]);
    //    img_0 = imread(samples::findFile(img_names_input[14]));

    //    LOGLN("Loading....................." << img_names_input[15]);
    //    img_1 = imread(samples::findFile(img_names_input[15]));

    //    //I_14_15
    //    partial_result = image_stitch(img_0, img_1);
    //           
    //    img_0.release();
    //    img_1.release();

    //    matsnd(partial_result, 3, world_rank);

    //    partial_result.release();
    //}



    /********************************************************************************
    *                                                                              *
    *                                  N=4                                         *
    *                                                                              *
    ********************************************************************************/
    //if (world_rank == 0) {

    //    img_0 = imread(samples::findFile(img_names_input[0]));
    //    img_1 = imread(samples::findFile(img_names_input[1]));

    //    // I_0_1
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0.release();
    //    img_1.release();

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[2]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[3]));
    //    partial_result = image_stitch(img_0, img_1);


    //    img_0 = partial_result;
    //    img_1 = matrcv(1, buffer, world_rank);
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = matrcv(2, buffer, world_rank);
    //    partial_result = image_stitch(img_0, img_1);

    //    imwrite(format("process_%d.png", world_rank), partial_result);

    //    partial_result.release();
    //}
    //if (world_rank == 1) {

    //    img_0 = imread(samples::findFile(img_names_input[4]));
    //    img_1 = imread(samples::findFile(img_names_input[5]));

    //    // I_0_1
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0.release();
    //    img_1.release();

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[6]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[7]));
    //    partial_result = image_stitch(img_0, img_1);

    //    matsnd(partial_result, 0, world_rank);

    //    partial_result.release();
    //}
    //if (world_rank == 2) {

    //    img_0 = imread(samples::findFile(img_names_input[8]));
    //    img_1 = imread(samples::findFile(img_names_input[9])); // add image 9 if GPU enabled

    //    // I_0_1
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0.release();
    //    img_1.release();

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[10]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[11]));
    //    partial_result = image_stitch(img_0, img_1);


    //    img_0 = partial_result;
    //    img_1 = matrcv(3, buffer, world_rank);
    //    partial_result = image_stitch(img_0, img_1);

    //    matsnd(partial_result, 0, world_rank);

    //    partial_result.release();
    //}
    //if (world_rank == 3) {

    //    img_0 = imread(samples::findFile(img_names_input[12]));
    //    img_1 = imread(samples::findFile(img_names_input[14]));

    //    // I_0_1
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0.release();
    //    img_1.release();

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[15]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[16]));
    //    partial_result = image_stitch(img_0, img_1);

    //    matsnd(partial_result, 2, world_rank);

    //    partial_result.release();
    //}

    /********************************************************************************
     *                                                                              *
     *                                  N=2                                         *
     *                                                                              *
    ********************************************************************************/
    //if (world_rank == 0) {

    //    img_0 = imread(samples::findFile(img_names_input[0]));
    //    img_1 = imread(samples::findFile(img_names_input[1]));

    //    // I_0_1
    //    partial_result = image_stitch(img_0, img_1);
    //    partial_result.convertTo(partial_result, CV_8UC1);

    //    img_0.release();
    //    img_1.release();

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[2]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[3]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[4]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[5]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[6]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[7]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = matrcv(1, buffer, world_rank);
    //    partial_result = image_stitch(img_0, img_1);
    //    imwrite(format("process_%d.png", world_rank), partial_result);

    //    partial_result.release();
    //}

    //if (world_rank == 1) {

    //    img_0 = imread(samples::findFile(img_names_input[8]));
    //    img_1 = imread(samples::findFile(img_names_input[10])); // add image 9 if GPU enabled

    //    //I_4_5
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0.release();
    //    img_1.release();

    //    //I_4_5_6_7
    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[10]));
    //    partial_result = image_stitch(img_0, img_1);

    //    //I_4_5_6_7
    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[11]));
    //    partial_result = image_stitch(img_0, img_1);

    //    //I_4_5_6_7
    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[12]));
    //    partial_result = image_stitch(img_0, img_1);

    //    //I_4_5_6_7
    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[13]));
    //    partial_result = image_stitch(img_0, img_1);

    //    //I_4_5_6_7
    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[14]));
    //    partial_result = image_stitch(img_0, img_1);

    //    //I_4_5_6_7
    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[15]));
    //    partial_result = image_stitch(img_0, img_1);

    //   matsnd(partial_result, 0, world_rank);

    //    partial_result.release();
    //}


    /********************************************************************************
     *                                                                              *
     *                                  N=1                                         *
     *                                                                              *
    ********************************************************************************/
    //if (world_rank == 0) {

    //    img_0 = imread(samples::findFile(img_names_input[0]));
    //    img_1 = imread(samples::findFile(img_names_input[1]));

    //    // I_0_1
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0.release();
    //    img_1.release();

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[2]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[3]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[4]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[5]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[6]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[7]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[8]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[9]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[10]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[11]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[12]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[13]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[14]));
    //    partial_result = image_stitch(img_0, img_1);

    //    img_0 = partial_result;
    //    img_1 = imread(samples::findFile(img_names_input[15]));
    //    partial_result = image_stitch(img_0, img_1);

    //    ////----------------------------------------------------------------------------
    //    //img_0 = partial_result;
    //    //img_1 = imread(samples::findFile(img_names_input[16]));
    //    //partial_result = image_stitch(img_0, img_1);

    //    //img_0 = partial_result;
    //    //img_1 = imread(samples::findFile(img_names_input[17]));
    //    //partial_result = image_stitch(img_0, img_1);

    //    //img_0 = partial_result;
    //    //img_1 = imread(samples::findFile(img_names_input[18]));
    //    //partial_result = image_stitch(img_0, img_1);

    //    //img_0 = partial_result;
    //    //img_1 = imread(samples::findFile(img_names_input[19]));
    //    //partial_result = image_stitch(img_0, img_1);

    //    //img_0 = partial_result;
    //    //img_1 = imread(samples::findFile(img_names_input[20]));
    //    //partial_result = image_stitch(img_0, img_1);

    //    //img_0 = partial_result;
    //    //img_1 = imread(samples::findFile(img_names_input[21]));
    //    //partial_result = image_stitch(img_0, img_1);

    //    //img_0 = partial_result;
    //    //img_1 = imread(samples::findFile(img_names_input[22]));
    //    //partial_result = image_stitch(img_0, img_1);

    //    //img_0 = partial_result;
    //    //img_1 = imread(samples::findFile(img_names_input[23]));
    //    //partial_result = image_stitch(img_0, img_1);

    //    //img_0 = partial_result;
    //    //img_1 = imread(samples::findFile(img_names_input[24]));
    //    //partial_result = image_stitch(img_0, img_1);

    //    //img_0 = partial_result;
    //    //img_1 = imread(samples::findFile(img_names_input[25]));
    //    //partial_result = image_stitch(img_0, img_1);


    //    imwrite(format("process_%d.png", world_rank), partial_result);

    //    partial_result.release();
    //}

    int num_procs = 2;
    int stride = 1;
    int transfer_index = 1;
    int s = (int)floor(num_images_input / num_procs); // inverval size
    int s0 = s + num_images_input % num_procs; // find starting point. process[0] will handle elements 0 to s0

    int startIndex = s0 + (world_rank - 1) * s;
    int endIndex = startIndex + s;

    vector<int> active_processors;
    vector<int> senders;
    vector<int> receivers;

    /////////////////////////////////////////////////////// serial code ///////////////////////////////////////////////////////
//    if (world_rank == 0) {

//         partial_result = imread(samples::findFile(img_names_input[0]));

//         for (int i = 0; i < num_images_input - 1; i++) {
//             img_0 = partial_result;
//             img_1 = imread(samples::findFile(img_names_input[i + 1]));

//             partial_result = image_stitch(img_0, img_1, world_rank, "hi");
//         }

//         imwrite(format("process_%d_serial_%d_imgs.png", world_rank, num_images_input), partial_result);

//     }

    /////////////////////////////////////////////////////// parallel code ///////////////////////////////////////////////////////

    for (int i = 0; i <= log2(num_procs); i++) {

        if (i == 0) {
            // populate active processors
            for (int j = 0; j < num_procs; j += stride) {
                active_processors.push_back(j);

                
                // initialize each with data
                if (world_rank == j) {
                    if (world_rank == 0) {
                        cout << "num_images_input................................................" << num_images_input << endl;
                        cout << "PROCESSOR............................. " << j << " STARTIDX: - 0 " << " ENDIDX - " << s0-1 << endl;

                        partial_result = imread(samples::findFile(img_names_input[0]));
                        for (int k = 0; k < s0-1; k++) {
                            img_0 = partial_result;

                            img_1 = imread(samples::findFile(img_names_input[(double)k + 1]));

                            partial_result = image_stitch(img_0, img_1, world_rank, img_names_input[(double)k + 1]);
                           // imwrite(format("process_%d_img__%d_img__%d.jpg", j, k, k+1), partial_result);
                        }
                    }
                    else {
                        cout << "PROCESSOR............................. " << j << " STARTIDX: - " << startIndex << " ENDIDX - " << endIndex-1 <<  endl;

                        partial_result = imread(samples::findFile(img_names_input[startIndex]));
                        for (int k = startIndex; k < endIndex-1 ; k++) {
                            img_0 = partial_result;

                            img_1 = imread(samples::findFile(img_names_input[(double)k + 1]));

                            partial_result = image_stitch(img_0, img_1, world_rank, img_names_input[(double)k + 1]);
                           // imwrite(format("process_%d_img__%d_img__%d.jpg", j, k, k+1), partial_result);
                        }
                    }
                }
            }
            i++;
        }
        else {
            for (int j = 0; j < num_procs; j += stride) {
                active_processors.push_back(j);
            }
        }


        // populate senders and receivers
        for (int k = 0; k < active_processors.size(); k++) {
            if (k % 2 == 0){
                receivers.push_back(active_processors[k]);
            }
            else {
                senders.push_back(active_processors[k]);
            }
        }


        // execute senders and receivers
        for (int k = 0; k < active_processors.size(); k++) {
            if (world_rank == active_processors[k]) {
                cout << "HERE" << endl;////////////////////////////////// probably blocking here and waiting for senders to finish...
                if (k % 2 != 0) {
                    cout << "process[" << world_rank << "] sending to process[" << world_rank - pow(2, transfer_index - 1) << "]" << endl;
                    matsnd(partial_result, world_rank - pow(2, transfer_index - 1), k);
                }
                else {
                    cout << "process[" << world_rank << "] receiving from process[" << world_rank + pow(2, transfer_index - 1) << "]" << endl;
                    img_0 = partial_result;
                    img_1 = matrcv(k, buffer, active_processors[k] + pow(2, transfer_index - 1));
                    partial_result = image_stitch(img_0, img_1, -1, "hello");
                   // imwrite(format("process_%d_receiving_from_process_%f_iteration_%d.jpg", world_rank, world_rank + pow(2, transfer_index - 1), i), partial_result);
                }
                // base case
                if (active_processors.size() == 2 && world_rank == 0) {
                    imwrite(format("process_%d_parallel_act_siz_%d.jpg", world_rank, active_processors.size()), partial_result);
                }
            }
            
        }

        /*cout << "Active processors: [" << i << "]";
        for (int i = 0; i < active_processors.size(); i++) {
            cout << " " << active_processors[i];
        }
        cout << endl;
        cout << "Senders: ";
        for (int i = 0; i < senders.size(); i++) {
            cout << " " << senders[i];
        }
        cout << endl;
        cout << "Receivers: ";
        for (int i = 0; i < receivers.size(); i++) {
            cout << " " << receivers[i];
        }
        cout << endl;*/

        active_processors.clear();
        senders.clear();
        receivers.clear();
        stride = 2 * stride;
        transfer_index++; 
        
    }
    

    MPI_Finalize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "process_" << world_rank << " ------------ Finished, total time : " << diff.count() << " s------------\n";
}

