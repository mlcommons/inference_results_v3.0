#ifndef __GST_MLPERF_SRC_H__
#define __GST_MLPERF_SRC_H__

#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>

G_BEGIN_DECLS
#define GST_TYPE_MLPERFSRC \
    (gst_mlperfsrc_get_type())
#define GST_MLPERFSRC(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_MLPERFSRC,GstMlperfSrc))
#define GST_MLPERFSRC_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_MLPERFSRC,GstMlperfSrcClass))
#define GST_IS_MLPERFSRC(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_MLPERFSRC))
#define GST_IS_MLPERFSRC_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_MLPERFSRC))
#define GST_MLPERFSRC_CAST(obj) ((GstMlperfSrc *) (obj))
typedef struct _GstMlperfSrc GstMlperfSrc;
typedef struct _GstMlperfSrcClass GstMlperfSrcClass;
typedef struct _GstMlperfSrcPrivate GstMlperfSrcPrivate;

/**
 * @brief GstMlperfSrc data structure.
 */
struct _GstMlperfSrc
{
  GstBaseSrc element;
  GAsyncQueue *data_queue;
  GstMlperfSrcPrivate * priv;
};

/**
 * @brief GstMlperfSrcClass data structure.
 */
struct _GstMlperfSrcClass
{
  GstBaseSrcClass parent_class;
};

GType gst_mlperfsrc_get_type (void);

G_END_DECLS
#endif /* __GST_MLPERF_SRC_H__ */
