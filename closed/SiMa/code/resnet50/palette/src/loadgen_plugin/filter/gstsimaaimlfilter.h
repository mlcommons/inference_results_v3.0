#ifndef __GST_ML_FILTER_H__
#define __GST_ML_FILTER_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <stdio.h>

G_BEGIN_DECLS

#define GST_TYPE_ML_FILTER             (gst_ml_filter_get_type())
#define GST_ML_FILTER(obj)             (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_ML_FILTER,GstMlFilter))
#define GST_ML_FILTER_CLASS(klass)     (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_ML_FILTER,GstMlFilterClass))
#define GST_IS_ML_FILTER(obj)          (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_ML_FILTER))
#define GST_IS_ML_FILTER_CLASS(klass)  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_ML_FILTER))

typedef struct _GstMlFilter      GstMlFilter;
typedef struct _GstMlFilterClass GstMlFilterClass;
typedef struct _GstMlFilterPrivate GstMlFilterPrivate;

struct _GstMlFilter {
  GstBaseTransform parent;
  GstMlFilterPrivate * priv;
};

struct _GstMlFilterClass {
  GstBaseTransformClass parent_class;
};

GType gst_ml_filter_get_type (void);

G_END_DECLS

#endif /* __GST_ML_FILTER_H__ */
