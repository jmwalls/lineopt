# LineOpt

We're interested in updating a set of line annotations given a set of keypoint
observations.

Line annotations and keypoint data is organized wrt the edges of a coarse graph.

* Parse all keypoint data into a dataframe per edge with the following
  attributes
    * type
    * position
    * range to vehicle
    * x (ecef)
    * y (ecef)
    * z (ecef)
    * geometry (Point lon/lat/alt)
* Parse all annotation data into a dataframe per edge with the following
  attributes
    * line id
    * lane group id
    * geometry (LineStr lon/lat/alt)
* Organize data into edges
    * Associate lane groups with edges
    * Associate keypoint data with edges
* Associate keypoint data to a particular line annotation
* Update line geometries given the set of observations associated to it.
