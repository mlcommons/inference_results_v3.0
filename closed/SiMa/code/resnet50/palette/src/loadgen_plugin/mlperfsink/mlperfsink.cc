#include <gst/gst.h>
#include "mlperfsink.h"

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifdef HAVE_UNISTD_H
#  include <unistd.h>
#endif

#include <errno.h>
#include <string.h>

#include "loadgen_wrapper_intf.h"

#define MLA_VALID_NDIM 1001

static GstStaticPadTemplate sinktemplate = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

#ifdef MLPERF_VERBOSE_BUILD
#define MLPERF_VERBOSE(fmt, ...) (g_warning(fmt, __VA_ARGS__));
#else
#define MLPERF_VERBOSE(fmt, ...) 
#endif


GST_DEBUG_CATEGORY_STATIC (gst_mlperfsink_debug);
#define GST_CAT_DEFAULT gst_mlperfsink_debug

LoadgenSingleton * loader;

// #define MLPERF_VERBOSE_BUILD TRUE

/* mlperfsink signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_LOCATION
};

static void gst_mlperfsink_finalize (GObject * object);

static void gst_mlperfsink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_mlperfsink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_mlperfsink_start (GstBaseSink * basesink);
static gboolean gst_mlperfsink_stop (GstBaseSink * basesink);

static gboolean gst_mlperfsink_is_seekable (GstBaseSink * sink);
static gboolean gst_mlperfsink_get_size (GstBaseSink * sink, guint64 * size);

static GstFlowReturn gst_mlperfsink_render (GstBaseSink * sink,
    GstBuffer * buffer);
static GstFlowReturn gst_mlperfsink_render_list (GstBaseSink * sink,
    GstBufferList * list);


//G_IMPLEMENT_INTERFACE (GST_TYPE_URI_HANDLER, gst_mlperfsink_uri_handler_init);
// GST_DEBUG_CATEGORY_INIT (gst_mlperfsink_debug, "mlperfsink", 0, "mplerfsink element");

GST_ELEMENT_REGISTER_DEFINE(mlperfsink, "mlperfsink", GST_RANK_NONE, GST_TYPE_MLPERFSINK);
G_DEFINE_TYPE (GstMLPerfSink, gst_mlperfsink, GST_TYPE_BASE_SINK);
// G_DEFINE_TYPE_WITH_CODE (GstMLPerfSink, gst_mlperfsink, GST_TYPE_BASE_SRC, _do_init);

static void
gst_mlperfsink_class_init (GstMLPerfSinkClass * klass)
{
//       ** ****     ** ** **********
//      /**/**/**   /**/**/////**/// 
//      /**/**//**  /**/**    /**    
//      /**/** //** /**/**    /**    
//      /**/**  //**/**/**    /**    
//      /**/**   //****/**    /**    
//      /**/**    //***/**    /**    
//      // //      /// //     //     
  g_warning ("Calling Init Function...");
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSinkClass *gstbasesink_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);
  gstbasesink_class = GST_BASE_SINK_CLASS (klass);

  GST_DEBUG_CATEGORY_INIT(gst_mlperfsink_debug, "mlperfsink", 0,
      "SiMa.ai Accelerator Filter"
  );
  
  gobject_class->set_property = gst_mlperfsink_set_property;
  gobject_class->get_property = gst_mlperfsink_get_property;
  gobject_class->finalize = gst_mlperfsink_finalize;

  gst_element_class_set_static_metadata (gstelement_class,
      "MLPerf LoadGen",
      "Source/File",
      "MLPerf Load Gen From a file",
      "Victor Bittorf <victor.bittorf@sima.ai>");
  gst_element_class_add_static_pad_template (gstelement_class, &sinktemplate);

  gstbasesink_class->start = GST_DEBUG_FUNCPTR (gst_mlperfsink_start);
  gstbasesink_class->stop = GST_DEBUG_FUNCPTR (gst_mlperfsink_stop);
  gstbasesink_class->render = GST_DEBUG_FUNCPTR (gst_mlperfsink_render);
  gstbasesink_class->render_list = GST_DEBUG_FUNCPTR (gst_mlperfsink_render_list);

  if (sizeof (off_t) < 8) {
    GST_LOG ("No large file support, sizeof (off_t) = %" G_GSIZE_FORMAT "!",
        sizeof (off_t));
  }
  loader = LoadgenSingleton::get_instance();
  g_warning ("Returning from Init Function...");
}

static void
gst_mlperfsink_init (GstMLPerfSink * sink)
{
  sink->filename = NULL;
}

static void
gst_mlperfsink_finalize (GObject * object)
{
  GstMLPerfSink *sink;

  sink = GST_MLPERFSINK (object);

  g_free (sink->filename);
  // g_free (sink->uri);

  // G_OBJECT_CLASS (parent_class)->finalize (object);
}

static void
gst_mlperfsink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstMLPerfSink *sink;

  g_return_if_fail (GST_IS_MLPERFSINK (object));

  sink = GST_MLPERFSINK (object);

  G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
}

static void
gst_mlperfsink_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstMLPerfSink *sink;

  g_return_if_fail (GST_IS_MLPERFSINK (object));

  sink = GST_MLPERFSINK (object);

  switch (prop_id) {
    case PROP_LOCATION:
      g_value_set_string (value, sink->filename);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static GstFlowReturn gst_mlperfsink_render (GstBaseSink * sink,
					    GstBuffer * buffer) {
    GstMLPerfSink *sink_;
    guint to_read, bytes_read;
    int ret;
    GstMapInfo info;
    guint8 *data;
      
    size_t new_offset = 0;
    size_t new_offset_end = 0; 

    GST_DEBUG("MLPerfSink: render(buffer->offset=%lld buffer->offset_end=%lld)", buffer->offset, buffer->offset_end);
    
    if (!gst_buffer_map (buffer, &info, GST_MAP_READ)) {
        GST_ELEMENT_ERROR (sink, RESOURCE, WRITE, (NULL), ("Can't read from buffer"));
        return GST_FLOW_ERROR;
    }

    
    size_t batch_size = loader->loadgen_get_cur_bs();
    size_t ndim_sz = 1008;
    size_t out_sz = batch_size * ndim_sz;

    // auto out = std::make_unique<int8_t[]>(out_sz);
    // std::memcpy(out.get(), (int8_t *)info.data, out_sz);

    // Compute top-1
    int8_t *classes = (int8_t*) info.data;
    std::vector<int32_t> arg_maxes;
    for (size_t img = 0; img < batch_size; img++) {
        size_t best_i = 0;
        int8_t best_val = classes[ndim_sz * img];
        for (size_t i = 0; i < MLA_VALID_NDIM; i++) {
            int8_t val = classes[ndim_sz * img + i];
            if (val > best_val) {
                best_i = i;
                best_val = val;
            }
        }
        GST_DEBUG("bs:%ld, image_id1 = %d, class_id = %d ",
		  loader->loadgen_get_cur_bs(),
                  loader->loadgen_get_current_sample_index(img),
                  (best_i - 1));
        arg_maxes.push_back(best_i - 1);
    }


    loader->loadgen_complete_sample(arg_maxes);
    gst_buffer_unmap(buffer, &info);
    // mlperf_example_complete(arg_maxes);
    return GST_FLOW_OK;
}

static GstFlowReturn gst_mlperfsink_render_list (GstBaseSink * sink,
						 GstBufferList * list) {
     // g_warning ("MLPerfSink: render_list()");
     return GST_FLOW_OK;
}

static gboolean
gst_mlperfsink_is_seekable (GstBaseSink * basesink)
{
  return false;
}

static gboolean
gst_mlperfsink_get_size (GstBaseSink * basesink, guint64 * size)
{
  struct stat stat_results;
  GstMLPerfSink *sink;

  sink = GST_MLPERFSINK (basesink);

  // if (!sink->seekable) {
    // /* If it isn't seekable, we won't know the length (but fstat will still
     // * succeed, and wrongly say our length is zero. */
    // return FALSE;
  // }

  // if (fstat (sink->fd, &stat_results) < 0)
  //  goto could_not_stat;

  *size = stat_results.st_size;

  return TRUE;

  /* ERROR */
could_not_stat:
  {
    return FALSE;
  }
}

/* open the file, necessary to go to READY state */
static gboolean
gst_mlperfsink_start (GstBaseSink * basesink)
{
  GstMLPerfSink *sink = GST_MLPERFSINK (basesink);

//      ******** **********     **     *******   **********
//     **////// /////**///     ****   /**////** /////**/// 
//    /**           /**       **//**  /**   /**     /**    
//    /*********    /**      **  //** /*******      /**    
//    ////////**    /**     **********/**///**      /**    
//           /**    /**    /**//////**/**  //**     /**    
//     ********     /**    /**     /**/**   //**    /**    
//    ////////      //     //      // //     //     //     
  
  struct stat stat_results;

  SiMaSUT simasut;
  SiMaQSL simaqsl;

  return TRUE;
}

/* unmap and close the file */
static gboolean
gst_mlperfsink_stop (GstBaseSink * basesink)
{
//        ******** **********   *******   ******* 
//       **////// /////**///   **/////** /**////**
//      /**           /**     **     //**/**   /**
//      /*********    /**    /**      /**/******* 
//      ////////**    /**    /**      /**/**////  
//             /**    /**    //**     ** /**      
//       ********     /**     //*******  /**      
//      ////////      //       ///////   //       

  GstMLPerfSink *sink = GST_MLPERFSINK (basesink);

  /* close the file */
  // close (sink->fd);

  /* zero out a lot of our state */
  // sink->fd = 0;
  // sink->is_regular = FALSE;

  return TRUE;
}

static gboolean
plugin_init (GstPlugin *plugin)
{
  return GST_ELEMENT_REGISTER (mlperfsink, plugin);
}

GST_PLUGIN_DEFINE (
  GST_VERSION_MAJOR,
  GST_VERSION_MINOR,
  simaaimlperfsink,
  "SiMa MLPerf Sink",
  plugin_init,
  "0.1",
  "unknown",
  "GStreamer",
  "SiMa.ai"
)
