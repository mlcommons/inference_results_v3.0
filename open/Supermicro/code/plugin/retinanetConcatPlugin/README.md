[BS, 1] [BS,1000,4] [BS,1000] [BS,1000] -> [BS,7001] (Duplicate one of the [BS,1000] twice)


inputs:
  * (A) : bbox       [BS, 1000, 4] : swap the [0,1] and [2,3] dim so we change [xmin, ymin, xmax, ymax] to [ymin, xmin, ymax, xmax]
  * (B) : score      [BS, 1000]    : unsqueezed into [BS, 1000, 1]
  * (C) : label      [BS, 1000]    : unsqueezed into [BS, 1000, 1]
  * (D) : keep_count [BS, 1]

1. Concat bbox,score,label in the order of [score, bbox, score, label] so they are together in one batch. [BS, 1000, 7]
2. Reshape to [BS, 7000], then concat with keep_count so it becomes [BS

 [ B ] 
 [ A ] interleaved
 [ B ]
 [ C ]
