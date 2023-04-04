#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <unistd.h>

#include <simaai/gst-api.h>
#include <simaai/gstsimaaiallocator.h>
#include <simaai/parser_types.h>
#include <simaai/parser.h>
#include <simaai/simaai_memory.h>
#include <simaai/gst-api.h>

#include "gstsimaaimlfilter.h"

#include "mla_rt_client.h"

GST_DEBUG_CATEGORY_STATIC(gst_ml_filter_debug);
#define GST_CAT_DEFAULT gst_ml_filter_debug

#include "loadgen_wrapper.h"

#define UNUSED(x) (void)(x)
#define MLPERF_DEFAULT_SIZE (1)
#define MLPERFSRC_MAX_IN_TOKENS 4 /* Used for dimension parsing */
#define MLPERFSRC_MAX_OUT_TOKENS 2 /* Used for dimension parsing */
#define CONFIG_FILE "config.json"
#define IN_TENSOR_TYPE (0)
#define OUT_TENSOR_TYPE (1)
#define TOY_MODE_NUM_OF_IMAGES 1000
#define MLA_VALID_NDIM 1001
// #define ML_FILTER_EVAL_PERF

/**
 * @brief Tensor Dimesnion 
 * outdimension :: dimdim[1]:dim[0]
 * BatchSize:MLA_Outsz
 * intensor dimesnion: dim[3]:dim[2]:dim[1]:dim[0]
 * N:C:H:W
 */
typedef struct TensorDim {
  gint32 dim[6]; ///< The NCHW representation of the tensor dimension
} TensorDim;

struct _GstMlFilterPrivate
{
  GstAllocator * allocator; ///< gst allocator handle 
  GstMemory * in_mem; ///< The opaque pointer to simaai memory
  GstMemory * out_mem; ///< The opaque pointer to simaai memory
  GThread * worker_thread; ///< worker thread to get data from mlperf imagenet
  GString * config_file;

  gint64 buf_id; ///< simaai memory buffer id, returned after allocation. Passed onto next plugin
  gint64 out_buf_id; ///< simaai memory buffer id, returned after allocation. Passed onto next plugin
  gchar *filename; ///< filename of where the input data is available
  guint64 scenario; ///< Scenario type, {0,1,2} {SingleStream, MultiStream, Offline}
  gboolean toy_mode; ///< Toy mode, runs with 10000 images
  gsize bytes_available; ///< The number of bytes read from imagenet. Should match tensor_dims
  gint64 frame_id; ///< The sequence id incremented for every iteration
  gint64 num_of_images; ///< Number of samples to use
  guint run_type; ////< 0=Performance or 1=Accuracy
    
  TensorDim in_tensor_dim; ///< Refer above the tensor dimesnion in NCHW format
  TensorDim out_tensor_dim; ///< Out tensor dimension

  simaai_params_t * params;///< SiMa.ai json parser handle
};

enum {
  PROP_0,
  PROP_CONF_F,
  PROP_LOCATION,
  PROP_TOY_MODE,
  PROP_SCENARIO,
  PROP_IN_TENSOR_DIMS,
  PROP_OUT_TENSOR_DIMS,
  PROP_NUM_OF_IMAGES,
  PROP_RUN_TYPE,
  
  PROP_LAST
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE (
  "sink",
  GST_PAD_SINK,
  GST_PAD_ALWAYS,
  GST_STATIC_CAPS_ANY
);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE (
  "src",
  GST_PAD_SRC,
  GST_PAD_ALWAYS,
  GST_STATIC_CAPS_ANY
);

#define gst_ml_filter_parent_class parent_class
G_DEFINE_TYPE(GstMlFilter, gst_ml_filter, GST_TYPE_BASE_TRANSFORM);

/**
 * \brief Loadgen Wrapper declaration, used by the plugin and handler
 */
simaai::mlperf_wrapper::LoadgenWrapper<int8_t> loader;

static gint
gst_ml_filter_validate_data_size (GstMlFilter * self, const gsize sz);

static gboolean
gst_ml_filter_set_tensor_dimension (GstMlFilter * self, const gchar * dimstr);

static gsize
gst_ml_filter_get_tensor_size (GstMlFilter * self, gint type);

static void
gst_ml_filter_complete_run (GstMlFilter * self);

/**
 * @brief Function to validate the size of the NCHW represented tensor
 *
 * @param self Opaque pointer to GstMlFilter gobject
 * @param sz to be validated
 * @return 1 on success 0 on failure 
 */
static gint
gst_ml_filter_validate_data_size (GstMlFilter * self, const gsize sz) {
  return ((((gint) sz % (gint) gst_ml_filter_get_tensor_size(self, IN_TENSOR_TYPE)) == 0) ? 1: 0);
}

/**
 * @brief API to setup TensorDim structure after parsing the params 
 *
 * @param self Opaque pointer to GstMlFilter gobject
 * @param dimstr the dimension string that is to be parsed in the "N:C:H:W" format
 * @return TRUE on success FALSE on failure
 */
static gboolean
gst_ml_filter_set_tensor_dimension (GstMlFilter * self, const gchar * dimstr, gint no_of_slices)
{
  guint rank = 0;
  guint64 val;
  gchar **strv;
  gchar *dim_string;
  guint i, num_dims;

  if (dimstr == NULL)
    return 0;

  /* remove spaces */
  dim_string = g_strdup (dimstr);
  g_strstrip (dim_string);

  strv = g_strsplit (dim_string, ":", MLPERFSRC_MAX_IN_TOKENS);
  num_dims = g_strv_length (strv);
  if (num_dims > no_of_slices) {
    GST_ERROR("Unsupported tensor dimension by SiMa MLA");
    return FALSE;
  }

  if (no_of_slices == 4) {
    for (i = 0; i < num_dims; i++) {
      g_strstrip (strv[i]);
      if (strv[i] == NULL || strlen (strv[i]) == 0)
        break;

      val = g_ascii_strtoull (strv[i], NULL, 10);
      self->priv->in_tensor_dim.dim[i] = (uint32_t) val;
      rank = i + 1;
    }

    for (; i < MLPERFSRC_MAX_IN_TOKENS; i++)
      self->priv->in_tensor_dim.dim[i] = 1;
  } else {
    for (i = 0; i < num_dims; i++) {
      g_strstrip (strv[i]);
      if (strv[i] == NULL || strlen (strv[i]) == 0)
        break;

      val = g_ascii_strtoull (strv[i], NULL, 10);
      self->priv->out_tensor_dim.dim[i] = (uint32_t) val;
      rank = i + 1;
    }

    for (; i < MLPERFSRC_MAX_OUT_TOKENS; i++)
      self->priv->out_tensor_dim.dim[i] = 1;
  }

  g_strfreev (strv);
  g_free (dim_string);
  return rank;
}

/**
 * @brief Returns the tensor size for memory allocation
 *
 * @param self Opaque pointer to GstMlFilter gobject
 * @return Size of tensor based on dims 
 */
static gsize
gst_ml_filter_get_tensor_size (GstMlFilter * self, gint type)
{
  gsize size = MLPERF_DEFAULT_SIZE;

  if (type == 0) {
    for (int i = 0; i < 4 ; i++)
      size *= self->priv->in_tensor_dim.dim[i];
  } else {
    for (int i = 0; i < 2 ; i++)
      size *= self->priv->out_tensor_dim.dim[i];
  }

  return size;
}

/**
 * @brief Gstreamer set property for the config values
 *
 * @param object input gobject
 * @param prop_id the property id as maintined in gstreamer
 * @param value the value for the property
 * @param psepc, property param specification, metadata of a proerty
 */
static void
gst_ml_filter_set_property (GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
  GstMlFilter *self = GST_ML_FILTER(object);

  switch (prop_id) {
    case PROP_CONF_F:
      g_string_assign(self->priv->config_file, g_value_get_string(value));
      GST_DEBUG_OBJECT(self, "Config argument was changed to %s", self->priv->config_file->str);
      break;
    case PROP_LOCATION:
      self->priv->filename = g_strdup (g_value_get_string(value));
      break;
    case PROP_TOY_MODE:
      self->priv->toy_mode = g_value_get_boolean (value);
      break;
    case PROP_SCENARIO:
      self->priv->scenario = g_value_get_uint (value);
      break;
    case PROP_RUN_TYPE:
      self->priv->run_type = g_value_get_uint (value);
      break;
    case PROP_NUM_OF_IMAGES:
      self->priv->num_of_images = g_value_get_int64 (value);
      break;
    case PROP_IN_TENSOR_DIMS:
      gst_ml_filter_set_tensor_dimension (self, g_value_get_string(value), 4);
      break;
    case PROP_OUT_TENSOR_DIMS:
      gst_ml_filter_set_tensor_dimension (self, g_value_get_string(value), 2);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Gstreamer get property for the config values
 *
 * @param object input gobject
 * @param prop_id the property id as maintined in gstreamer
 * @param value the value for the property
 * @param psepc, property param specification, metadata of a proerty
 */
static void
gst_ml_filter_get_property (GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
  GstMlFilter *self = GST_ML_FILTER(object);

  switch (prop_id) {
  case PROP_LOCATION:
       g_value_set_string (value, self->priv->filename);
       break;
  case PROP_CONF_F:
       g_value_set_string(value, (const gchar *)self->priv->config_file);
       break;
  default:
       G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
       break;
  }
}

/** 
 * @brief Creates a mlpef load test and logging configuration
 * @param self GstMlFilter reference object
 * @param LoadgenConfig configuration object
 */
static void
gst_ml_filter_get_loadgen_config (GstMlFilter * self, simaai::mlperf_wrapper::LoadgenConfig & cfg) {
  cfg.test_type = static_cast<simaai::mlperf_wrapper::MlperfTestType>(self->priv->scenario);
  cfg.run_type = static_cast<simaai::mlperf_wrapper::MlperfRunType>(self->priv->run_type);
  cfg.sample_data_fpath = self->priv->filename;
  cfg.cur_batch_size = self->priv->in_tensor_dim.dim[0];
  cfg.cur_buffer_size = gst_ml_filter_get_tensor_size(self, IN_TENSOR_TYPE);
  cfg.num_of_images = (self->priv->toy_mode == TRUE ? 1000 : 50000);

  cfg.dims.stride = self->priv->in_tensor_dim.dim[1];
  cfg.dims.width = self->priv->in_tensor_dim.dim[2];
  cfg.dims.height = self->priv->in_tensor_dim.dim[3];

  if (cfg.test_type == simaai::mlperf_wrapper::MlperfTestType::MULTI_STREAM_MODE)
      cfg.mlperf_test_settings.min_duration_ms = 600000;
  cfg.mlperf_test_settings.min_query_count = cfg.num_of_images;

  char * mlperf_conf_path = ((char *)parser_get_string(self->priv->params, "mlperf_conf"));
  if (!mlperf_conf_path) {
      GST_ERROR("mlperf confg path not found in the config using defaults");
      cfg.mlperf_conf_fpath = "./mlperf_cfg/mlperf.conf";
  } else {
      cfg.mlperf_conf_fpath = mlperf_conf_path;

  }

  char * user_conf_path = ((char *)parser_get_string(self->priv->params, "user_conf"));
  if (!user_conf_path) {
      GST_ERROR("mlperf user config path not found in the config using defaults");
      cfg.user_conf_fpath = "./mlperf_cfg/user" + std::to_string(cfg.cur_batch_size) + ".conf";
  } else {
      cfg.user_conf_fpath = user_conf_path;
  }
  cfg.model_name = "resnet50";
}

/**
 * @brief Callback called when element starts processing
 *
 * @param trans the gobject for the base class GstBaseTransform
 * @return TRUE on success or FALSE on Failure
 */
static gboolean
gst_ml_filter_start (GstBaseTransform * trans)
{
  GstMlFilter *self = GST_ML_FILTER(trans);
  gsize tensor_size = gst_ml_filter_get_tensor_size(self, IN_TENSOR_TYPE);

  self->priv->params = parser_node_struct_init();
  if (self->priv->params == NULL) {
    GST_ERROR_OBJECT (self, "Unable to initialize parameters for node, file:%s", self->priv->config_file->str);
    return FALSE;
  }

  if ((parse_json_file(self->priv->config_file->str, self->priv->params)) != PARSER_SUCCESS) {
    GST_ERROR_OBJECT (self, "Unable to start parser");
    return FALSE;
  }
  
  size_t out_sz = gst_ml_filter_get_tensor_size(self, OUT_TENSOR_TYPE);
  self->priv->out_mem = simaai_target_mem_alloc_flags (SIMAAI_MEM_TARGET_GENERIC,
                                                       out_sz,
                                                       &self->priv->out_buf_id,
                                                       SIMAAI_MEM_FLAG_CACHED);

  if (self->priv->out_mem == NULL) {
    GST_ERROR_OBJECT(self, "ERROR: allocating contiguous memory for out_buf of size ");
    g_assert(self->priv->in_mem == NULL);
  }

  GST_DEBUG_OBJECT(self, "IN: Buffer id = %d, Out: Buffer id = %d", self->priv->buf_id, self->priv->out_buf_id);

  simaai::mlperf_wrapper::LoadgenConfig cfg;
  gst_ml_filter_get_loadgen_config(self, cfg);

  loader = simaai::mlperf_wrapper::LoadgenWrapper<int8_t>(cfg);
  loader.loadgen_start();

  char * model_path = ((char *)parser_get_string(self->priv->params, "model_path"));
  if (model_path == NULL) {
    GST_ERROR_OBJECT(self, "Model Path is null please check JSON config");
    return FALSE;
  }

  if (mla_rt_client_init(model_path) != 0) {
    GST_ERROR_OBJECT(self, "Failed to initialize dispatcher platform");
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Callback called when element stops processing
 *
 * @param trans the gobject for the base class GstBaseTransform
 * @return TRUE on success or FALSE on Failure
 */
static gboolean
gst_ml_filter_stop (GstBaseTransform *trans)
{
  GstMlFilter *self = GST_ML_FILTER(trans);
  return TRUE;
}

/**
 * @brief Virtual call to do transformation of input buffer to an output buffer, 
 * this is not inplace
 *
 * @param trans the gobject for the base class GstBaseTransform
 * @param inbuf input buffer to work on
 * @param outbuf output buffer to push
 * @return GST_FLOW_OK on success or GST_FLOW_ERROR on failure
 */
static GstFlowReturn
gst_ml_filter_transform (GstBaseTransform *trans, GstBuffer *inbuf,
			GstBuffer *outbuf)
{
  GstMlFilter *self = GST_ML_FILTER(trans);
  GstBuffer *buffer = NULL;
  int ret;

  while (true) {

#ifdef ML_FILTER_EVAL_PERF
    struct timeval start, end;
    long seconds, micros;
    gettimeofday(&start, NULL);
#endif
    // GstMapInfo map;
    // gst_memory_map(self->priv->in_mem, &map, GST_MAP_WRITE);

    size_t batch_size = self->priv->in_tensor_dim.dim[0];
    uint64_t  * phys_addr_arr = new uint64_t[batch_size];
    // std::uintptr_t addr_list_ptr = reinterpret_cast<std::uintptr_t>(phys_addr_arr);

    loader.loadgen_get_sample(&self->priv->bytes_available, phys_addr_arr);
    if (self->priv->bytes_available <= 0) {
      GST_INFO_OBJECT(self, "Sending EOS - Downstream");
      return GST_FLOW_EOS;
    }

    gsize data_len = self->priv->bytes_available;
    
    if (!gst_ml_filter_validate_data_size(self, data_len)) {
      g_error("Tensor dimension does'nt match the data size read");
      g_assert(self->priv->bytes_available);
    }

    if (data_len <= 0) {
      GST_ERROR_OBJECT(self, "data is invalid, size returned : [%d]", self->priv->bytes_available);
      return GST_FLOW_ERROR;
    }

    size_t in_img_sz = gst_ml_filter_get_tensor_size(self, IN_TENSOR_TYPE) / batch_size;
    size_t out_img_sz = gst_ml_filter_get_tensor_size(self, OUT_TENSOR_TYPE) / batch_size;
    
    if (mla_rt_client_batch_run_model(phys_addr_arr,
                                      self->priv->out_buf_id,
                                      0,
                                      batch_size,
                                      in_img_sz,
                                      out_img_sz) != 0) {
      GST_ERROR_OBJECT (self, "Failed to get buffer to push to the mlfilter.");
      delete [] phys_addr_arr;
      return GST_FLOW_ERROR;
    }

    simaai_memory_invalidate_cache(simaai_get_memory_handle(self->priv->out_mem));
    
    gst_ml_filter_complete_run (self);
#ifdef ML_FILTER_EVAL_PERF
    gettimeofday(&end, NULL);

    seconds = (end.tv_sec - start.tv_sec);
    micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("[%d]:[%s] runtime %ld secs, %ld micros\n", __LINE__, __func__, seconds , micros);
#endif
  }

  return GST_FLOW_OK;
}

/**
 * @brief Callback to finalize gboject
 *
 * @param object The object to be finalized
 */
static void
gst_ml_filter_finalize(GObject *object)
{
  GstMlFilter *self = GST_ML_FILTER(object);

  g_string_free(self->priv->config_file, TRUE);
  g_free (self->priv->filename);

  mla_rt_client_cleanup();

  simaai_memory_t * out_mem = simaai_get_memory_handle(self->priv->out_mem);
  simaai_memory_unmap(out_mem);
  simaai_memory_free(out_mem);
}

static int gst_ml_filter_run_topk (GstMlFilter * self, const int8_t * data) {
  // Compute top1
  size_t batch_size = self->priv->in_tensor_dim.dim[0];
  size_t ndim = self->priv->out_tensor_dim.dim[1];

  int8_t * classes = (int8_t *) data;
  std::vector<int32_t> arg_maxes;

  for (size_t img = 0; img < batch_size; img++) {
    size_t best_i = 0;
    std::vector<size_t> best_indexes;
    best_indexes.push_back(0);
    int8_t best_val = classes[ndim * img];
    for (size_t i = 0; i < MLA_VALID_NDIM ; i++) {
      int8_t val = classes[ndim * img + i];
      if (val >= best_val) {
        if (best_val != val) {
          best_indexes.clear();
        }
        best_i = i;
        best_indexes.push_back(i);
        best_val = val;
      }
    }

    best_i = (((best_indexes.size() > 1) ? best_indexes[1] : best_indexes[0]) -  1);
    GST_DEBUG_OBJECT(self, "seq_id = %d, image_id1 = %d, class_id = %d ",
                     self->priv->frame_id,
                     (loader.get_current_sample_index(img) + 1),
                     (best_i));
    arg_maxes.push_back(best_i);
  }
  loader.loadgen_complete_sample(arg_maxes);
  return 0;
}

/**
 * @brief After running a model on the mla we call this to complete the mlperf test 
 *
 * @param self An opaque pointer to GstMlFilter gobject
 * @param out_buf_id used to map the output buffer
 */
static void gst_ml_filter_complete_run (GstMlFilter * self)
{
  GstMapInfo map;

#ifdef ML_FILTER_EVAL_PERF
  struct timeval start, end;
  long seconds, micros;
  gettimeofday(&start, NULL);
#endif

  size_t out_sz = gst_ml_filter_get_tensor_size(self, OUT_TENSOR_TYPE);
  gst_memory_map(self->priv->out_mem, &map, GST_MAP_READ);

  int dump_data = *((int *)parser_get_int(self->priv->params, "dump_data"));
  if (dump_data) {
    if (mla_rt_client_write_output_buffer(map.data,
                                          out_sz,
                                          self->priv->frame_id) != 0) {
      GST_ERROR ("Failed to write output buffer");
      return ;
    }
  }

  size_t ndim = self->priv->out_tensor_dim.dim[1];
  int iter_sz = self->priv->in_tensor_dim.dim[0] / 4;
  
  gst_ml_filter_run_topk (self, (int8_t *)map.data);
  gst_memory_unmap(self->priv->out_mem, &map);

#ifdef ML_FILTER_EVAL_PERF
  gettimeofday(&end, NULL);

  seconds = (end.tv_sec - start.tv_sec);
  micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
  printf("[%d]:[%s] runtime %ld secs, %ld micros\n", __LINE__, __func__, seconds , micros);
#endif

  self->priv->frame_id++;

  return;
}

/**
 * @brief Called when class init is scheduled used to initialize the output buffer to be used by transform
 *
 * @param trans is a gobject of type GstBaseTransform
 * @param input input buffer
 * @param outbuf output allocated buffer
 */
static GstFlowReturn
gst_ml_filter_prepare_output_buffer (GstBaseTransform *trans, GstBuffer *input,
    GstBuffer **outbuf)
{
  GstBuffer *buf;

  *outbuf = NULL;

  buf = gst_buffer_new();
  if (buf) {
    *outbuf = buf;
  } else {
    GST_WARNING("Failed to allocate buffer");
    return GST_FLOW_ERROR;
  }

  return GST_FLOW_OK;
}

/**
 * @brief Callback to initialize the class
 *
 * @param klass class pointer
 */
static void
gst_ml_filter_class_init(GstMlFilterClass *klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  gobject_class = G_OBJECT_CLASS(klass);
  gstelement_class = GST_ELEMENT_CLASS(klass);
  gstbasetransform_class = GST_BASE_TRANSFORM_CLASS(klass);

  GST_DEBUG_CATEGORY_INIT(gst_ml_filter_debug, "ml_filter", 0,
      "SiMa.ai Accelerator Filter"
  );

  gobject_class->finalize = gst_ml_filter_finalize;
  gobject_class->set_property = gst_ml_filter_set_property;
  gobject_class->get_property = gst_ml_filter_get_property;

  g_object_class_install_property (gobject_class, PROP_CONF_F,
                                   g_param_spec_string ("config",
                                                        "ConfigFile",
                                                        "Config JSON to be used",
                                                        CONFIG_FILE,
                                                        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_LOCATION,
                                   g_param_spec_string ("inpath", "File Path",
                                                        "Input dat to read", NULL,
                                                        GParamFlags(G_PARAM_READWRITE |
                                                                    G_PARAM_STATIC_STRINGS |
                                                                    GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_SCENARIO,
                                   g_param_spec_uint ("mlperf-scenario", "Scenario",
                                                      "0 = SingleStream, 1 = MultiStream, 2 = Offline",
                                                      0, 3, 0,
                                                      GParamFlags(G_PARAM_READWRITE |
                                                                  GST_PARAM_MUTABLE_PLAYING |
                                                                  G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_RUN_TYPE,
                                   g_param_spec_uint ("mlperf-run-type", "Run Type",
                                                      "0=Perf, 1=Acc, 2=Power",
                                                      0, 3, 0,
                                                      GParamFlags(G_PARAM_READWRITE |
                                                                  GST_PARAM_MUTABLE_PLAYING |
                                                                  G_PARAM_STATIC_STRINGS)));
  
  g_object_class_install_property (gobject_class, PROP_NUM_OF_IMAGES,
                                   g_param_spec_int64 ("no-of-images", "num of images",
                                                      "Number of image samples to be used",
                                                      0, 50000, 0,
                                                      GParamFlags(G_PARAM_READWRITE |
                                                                  GST_PARAM_MUTABLE_PLAYING |
                                                                  G_PARAM_STATIC_STRINGS)));
  
  g_object_class_install_property (gobject_class, PROP_TOY_MODE,
                                   g_param_spec_boolean ("toy-mode", "Enable Toy Mode",
                                                         "Toy Mode uses fewer images to run faster and easier",
                                                         false, GParamFlags(G_PARAM_READWRITE |
                                                                            GST_PARAM_MUTABLE_PLAYING |
                                                                            G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_IN_TENSOR_DIMS,
                                   g_param_spec_string ("in-dims", "Tensor Dimension as N:C:H:W",
                                                        "Tensor Dimension as BatchSize:ChannelWidth:Height:Width", NULL,
                                                        GParamFlags(G_PARAM_READWRITE |
                                                                    G_PARAM_STATIC_STRINGS |
                                                                    GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OUT_TENSOR_DIMS,
                                   g_param_spec_string ("out-dims", "Tensor Dimension as N:SZ",
                                                        "Tensor Dimension as BatchSize:MLA_OutSize", NULL,
                                                        GParamFlags(G_PARAM_READWRITE |
                                                                    G_PARAM_STATIC_STRINGS |
                                                                    GST_PARAM_MUTABLE_READY)));
  
  gst_element_class_set_static_metadata(gstelement_class,
      "SiMa.ai ML-Filter element",
      "Filter/Image",
      "Communicates to MLA",
      "SiMa.ai"
  );

  gst_element_class_add_static_pad_template(gstelement_class, &src_template);
  gst_element_class_add_static_pad_template(gstelement_class, &sink_template);

  gstbasetransform_class->start = GST_DEBUG_FUNCPTR(gst_ml_filter_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_ml_filter_stop);
  gstbasetransform_class->prepare_output_buffer =
      GST_DEBUG_FUNCPTR(gst_ml_filter_prepare_output_buffer);
  gstbasetransform_class->transform = GST_DEBUG_FUNCPTR(gst_ml_filter_transform);

  static const gchar *tags[] = { NULL };
  gst_meta_register_custom ("GstSimaMeta", tags, NULL, NULL, NULL);

}

/**
 * @brief Subclass initialization
 *
 * @param self is a gobject of type GstMlFilter
 */
static void
gst_ml_filter_init (GstMlFilter *self)
{
  gst_simaai_buffer_memory_init_once();
  self->priv = (GstMlFilterPrivate *) g_malloc(sizeof(GstMlFilterPrivate));

  self->priv->filename = NULL;
  self->priv->frame_id = 1;
  self->priv->buf_id = -1;
  self->priv->num_of_images = 0;
  self->priv->run_type = 0;

  self->priv->out_buf_id = 0;
  self->priv->in_mem = 0;
  self->priv->out_mem = 0;

  self->priv->config_file = g_string_new(CONFIG_FILE);
}

/**
 * @brief Plugin initialization routine
 *
 * @param plugin plugin gst object
 */
static gboolean
plugin_init (GstPlugin *plugin)
{
  g_message ("MLFILTER: Plugin Init");
  if (!gst_element_register(plugin, "ml_filter", GST_RANK_NONE,
                            GST_TYPE_ML_FILTER)) {
    GST_ERROR("Unable to register ml_filter plugin");
    return FALSE;
  }

  return TRUE;
}

GST_PLUGIN_DEFINE (
  GST_VERSION_MAJOR,
  GST_VERSION_MINOR,
  simaaimlfilter,
  "SiMa MLPerf Src",
  plugin_init,
  "0.1",
  "unknown",
  "GStreamer",
  "SiMa.ai"
)
