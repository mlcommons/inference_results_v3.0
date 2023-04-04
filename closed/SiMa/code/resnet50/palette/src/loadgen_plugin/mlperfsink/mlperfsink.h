
#ifndef __GST_MLPERF_SRC_H__
#define __GST_MLPERF_SRC_H__

#include <sys/types.h>

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>


#define _GST_ELEMENT_REGISTER_DEFINE_BEGIN(element) \
G_BEGIN_DECLS \
gboolean G_PASTE (gst_element_register_, element) (GstPlugin * plugin) \
{ \
  {

/**
 * _GST_ELEMENT_REGISTER_DEFINE_END: (attributes doc.skip=true)
 */
#define _GST_ELEMENT_REGISTER_DEFINE_END(element_name, rank, type) \
  } \
  return gst_element_register (plugin, element_name, rank, type); \
} \
G_END_DECLS

#define GST_ELEMENT_REGISTER_DECLARE(element) \
G_BEGIN_DECLS \
gboolean G_PASTE(gst_element_register_, element) (GstPlugin * plugin); \
G_END_DECLS

#define GST_ELEMENT_REGISTER_DEFINE(e, e_n, r, t) _GST_ELEMENT_REGISTER_DEFINE_BEGIN(e) _GST_ELEMENT_REGISTER_DEFINE_END(e_n, r, t)


#define GST_ELEMENT_REGISTER(element, plugin) G_PASTE(gst_element_register_, element) (plugin)


G_BEGIN_DECLS

#define GST_TYPE_MLPERFSINK \
  (gst_mlperfsink_get_type())
#define GST_MLPERFSINK(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_MLPERFSINK, GstMLPerfSink))
#define GST_MLPERFSINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_MLPERFSINK, GstMLPerfSinkClass))
#define GST_IS_MLPERFSINK(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_MLPERFSINK))
#define GST_IS_MLPERFSINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_MLPERFSINK))
#define GST_MLPERFSINK_CAST(obj) ((GstMLPerfSink*) obj)

typedef struct _GstMLPerfSink GstMLPerfSink;
typedef struct _GstMLPerfSinkClass GstMLPerfSinkClass;

/**
 * Gstmlperfsink:
 *
 * Opaque #Gstmlperfsink structure.
 */
struct _GstMLPerfSink {
  GstBaseSink element;

  /*< private >*/
  gchar *filename;			/* filename */
  gchar *mode;			/* filename */
};

struct _GstMLPerfSinkClass {
  GstBaseSinkClass parent_class;
};

G_GNUC_INTERNAL GType gst_mlperfsink_get_type (void);

GST_ELEMENT_REGISTER_DECLARE(mlperfsink)

G_END_DECLS

#endif /* __GST_MLPERF_SRC_H__ */
