# PathSnap - Geographical checks and route limitations

## Section 5: Geographical data

**Projection used:** UTM, each trip in its own zone (25 different zones across the dataset; see `projection` column in trip_geo.csv).  
Worst projected-vs-haversine error over all trips: **0.553%**.  
Euclidean distance on raw degrees (x 111 km): median error **26.8%**, worst **110.8%** (grows with latitude, see plots/distance_method_error.png).  
Coordinate order (lon, lat) confirmed for **90/90** trips (median GPS-to-nearest-road distance 4.4 m).

## Section 6: GPS vs reference route

Distance from each GPS point to the nearest arc of its reference route, train trips only:

|       |    metres |
|:------|----------:|
| count | 208417    |
| mean  |     21.43 |
| std   |     46.25 |
| min   |      0    |
| 50%   |      5.61 |
| 90%   |     74.73 |
| 95%   |    140.27 |
| 99%   |    203.67 |
| max   |    569.66 |

**Emission sigma estimate (Newson & Krumm, 1.4826 x median, train only): 8.32 m**

**Stationary periods** (points staying inside a 30 m box for 60 s): 51 trips have some; median 0.4% of points, max 58.0%. Among trips with far-from-route points, 2 have mostly stationary far points (parked off the network).

| part         |     trip |   pct_stationary |   path_km_all |   path_km_moving |   rt_dist_median_m |   rt_dist_median_moving_m |
|:-------------|---------:|-----------------:|--------------:|-----------------:|-------------------:|--------------------------:|
| train        | 00000033 |             58   |          47.9 |             23.1 |              122.4 |                      32.2 |
| train        | 00000054 |             46   |         169.6 |            103.9 |              138.6 |                     128.1 |
| train        | 00000010 |             27.6 |          25.9 |             20.6 |                4.8 |                       5.4 |
| train        | 00000016 |             21   |          16.2 |             14.2 |                6.7 |                       8   |
| train        | 00000072 |             18.6 |          11.3 |              9.9 |                5.6 |                       5.5 |
| student_test | 00000077 |             18.3 |          36.4 |             31.1 |                6.1 |                       5.7 |
| train        | 00000096 |             16.2 |          71.7 |             67.8 |                4.4 |                       4.4 |
| train        | 00000084 |             14.3 |         136.8 |            119.8 |                6.5 |                       6   |

Points whose nearest road is not their route road (by >10 m): median 1.1% per trip, max 85.6%.

### Trips where the route covers the drive only partly (3)

Criteria: < 90% of route arcs have a GPS point within 50 m, or > 5% of GPS points are > 100 m from the route.

| part   |     trip |   n_gps |   rt_dist_median_m |   rt_dist_p95_m |   pct_gps_gt100m_from_route |   pct_route_arcs_covered_50m |   longest_far_run_pts |   pct_far_pts_stationary |   first_gps_on_route_idx |   last_gps_on_route_idx |
|:-------|---------:|--------:|-------------------:|----------------:|----------------------------:|-----------------------------:|----------------------:|-------------------------:|-------------------------:|------------------------:|
| train  | 00000002 |    7243 |                5.7 |           242.4 |                         9.9 |                        100   |                   513 |                      3.8 |                        0 |                    6729 |
| train  | 00000033 |    5844 |              122.4 |           142.1 |                        51   |                        100   |                  2977 |                     72   |                        0 |                    5843 |
| train  | 00000054 |   19313 |              138.6 |           210.1 |                        83.7 |                         99.3 |                  3218 |                     51.7 |                        0 |                   19312 |
