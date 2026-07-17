Lung cancer is the number one cause of cancer death worldwide. Not one of the leading causes — the leading cause,
in Algeria, lung cancer is the most prevalent cancer among men, accounting for 14.7% of all male cancer cases.

And here is the key insight: the challenge is not treatment. Medicine has treatments. The challenge is finding the cancer early enough that treatment can actually work.

according to SEER-based statistics used by the American Cancer Society and World Health Organization aligned reports,  when lung cancer is diagnosed at a localized stage, the five-year survival rate can reach approximately 65 percent. However, when diagnosed at an advanced stage, it drops to around 10 percent. This large gap highlights the critical importance of early detection.
Precise, automatic segmentation of pulmonary nodules is what makes early detection scalable. That is why this problem matters — and that is why we chose to work on it.
So how is lung cancer detected? Through CT scans , computed tomography. A CT scan of the chest is not a single image. It is a stack of over 300 cross-sectional slices of the lungs, each one a few millimetres thick. Reviewing all of them, one by one, looking for a tiny abnormality -- that is what radiologists currently do.
Before classifying whether a nodule is malignant, radiologists must carefully analyze its characteristics : shape , size and density, 
In practice, this is a time-consuming process because a single CT scan contains hundreds of slices per patient, and each scan must be reviewed carefully. In a clinical setting with many patients per day, this becomes susceptible to fatigue-related oversight. This motivates the need for automated segmentation systems that can assist radiologists by identifying and outlining nodules more efficiently,
Automated segmentation is the answer. Before a nodule can be diagnosed, it must be precisely outlined and isolated. Segmentation is the step that does exactly that: it draws a boundary around the nodule, separating it from everything else. It is the essential bridge between a raw CT scan and a clinical diagnosis.
This brings us to a fundamental technical choice: should we process CT scans in 2D, slice by slice, or in full 3D?
A 2D approach looks at one slice at a time. But a nodule is a three-dimensional object -------- it extends across multiple slices. Looking at it only in the axial view misses context that exists in the sagittal and coronal planes.
