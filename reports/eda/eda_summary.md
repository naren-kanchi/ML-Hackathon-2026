# PathSnap - Dataset exploration summary

## 1. Trips per partition

| part         |   trips |
|:-------------|--------:|
| student_test |      10 |
| train        |      80 |

## 2. GPS observations per trip

|       |    mean |   min |    50% |   max |
|:------|--------:|------:|-------:|------:|
| n_gps | 2528.63 |   262 | 1496.5 | 19313 |

## 3. File availability and sizes

| file   |   present |   total |   size_mb_mean |   size_mb_max |
|:-------|----------:|--------:|---------------:|--------------:|
| arcs   |        90 |      90 |          0.513 |         3.857 |
| nodes  |        90 |      90 |          0.447 |         2.945 |
| route  |        90 |      90 |          0.002 |         0.012 |
| track  |        90 |      90 |          0.083 |         0.626 |

## 4. Data types and column names

| file   | columns                          | expected_dtype   | cols_ok   | dtype_ok   |   non_numeric |
|:-------|:---------------------------------|:-----------------|:----------|:-----------|--------------:|
| track  | longitude, latitude, timestamp_s | float            | True      | True       |             0 |
| nodes  | longitude, latitude              | float            | True      | True       |             0 |
| arcs   | from_node_id, to_node_id         | int              | True      | True       |             0 |
| route  | arc_id                           | int              | True      | True       |             0 |

## 5. Missing / invalid GPS observations

|                         |   total across trips |
|:------------------------|---------------------:|
| gps_nan_rows            |                    0 |
| gps_out_of_range        |                    0 |
| gps_null_island         |                    0 |
| gps_dup_consecutive_pos |                    6 |
| gps_dup_timestamps      |                    0 |

## 6. Timestamp ordering and irregular gaps

Trips with non-monotonic timestamps: 0  
Trips with gaps >10s: 42  
Trips with gaps >60s: 23

|            |    mean |    min |     50% |    max |
|:-----------|--------:|-------:|--------:|-------:|
| dt_median  |    1    |   1    |    1    |      1 |
| pct_dt_1s  |   94.32 |  53.07 |   99.14 |    100 |
| max_gap_s  | 2408.52 |   1    |    7    | 153064 |
| duration_s | 5274.25 | 265    | 1710    | 154030 |

## 7. Geographical extent

|              |   mean |   min |   50% |    max |
|:-------------|-------:|------:|------:|-------:|
| extent_ew_km |  13.55 |  0.85 |  6.73 |  70.39 |
| extent_ns_km |  11.68 |  0.87 |  5.25 |  75.4  |
| path_km      |  36.39 |  5.55 | 24.07 | 169.25 |

## 8. Road-network size

|                 |     mean |    min |      50% |       max |
|:----------------|---------:|-------:|---------:|----------:|
| n_nodes         | 22490.8  | 299    |  9305    | 155003    |
| n_arcs          | 45132.7  | 649    | 18761.5  | 307392    |
| mean_out_degree |     2.02 |   1.67 |     2.03 |      2.37 |

## 9-10. Connectivity, isolated nodes, disconnected components

|                     |   mean |   min |    50% |     max |
|:--------------------|-------:|------:|-------:|--------:|
| isolated_nodes      |   0    |  0    |   0    |    0    |
| n_weak_components   |  37.2  |  1    |  19    |  421    |
| largest_wcc_pct     |  96.84 | 69.28 |  98.02 |  100    |
| n_strong_components | 233.47 |  2    | 107    | 1942    |
| largest_scc_pct     |  94.88 | 63.59 |  96.42 |   99.96 |
| n_singleton_scc     | 191.8  |  0    |  74.5  | 1750    |
| pct_bidirectional   |  92.23 | 35.13 |  95.11 |  100    |
| self_loops          |   0.27 |  0    |   0    |    8    |
| duplicate_arcs      |  21.14 |  0    |   2.5  |  322    |

## 11. Trips with large gaps or other anomalies (57 trips)

Thresholds: gap >60s, implied speed >250 km/h.

| part         |     trip | anomalies                                                           |
|:-------------|---------:|:--------------------------------------------------------------------|
| student_test | 00000000 | fragmented network (largest SCC 87%)                                |
| student_test | 00000040 | implausible jumps (max 262 km/h)                                    |
| student_test | 00000052 | fragmented network (largest SCC 82%)                                |
| student_test | 00000056 | route revisits arcs: 255                                            |
| student_test | 00000065 | gap >60s (max 210s)                                                 |
| student_test | 00000076 | route breaks: 1                                                     |
| train        | 00000001 | implausible jumps (max 261 km/h)                                    |
| train        | 00000002 | gap >60s (max 160s); route breaks: 1                                |
| train        | 00000005 | fragmented network (largest SCC 64%)                                |
| train        | 00000006 | gap >60s (max 301s); track starts at t=693                          |
| train        | 00000008 | implausible jumps (max 253 km/h); route revisits arcs: 1            |
| train        | 00000010 | fragmented network (largest SCC 86%)                                |
| train        | 00000011 | gap >60s (max 909s)                                                 |
| train        | 00000013 | implausible jumps (max 510 km/h)                                    |
| train        | 00000014 | gap >60s (max 72s); track starts at t=2                             |
| train        | 00000016 | fragmented network (largest SCC 86%)                                |
| train        | 00000018 | route revisits arcs: 9                                              |
| train        | 00000019 | route breaks: 3                                                     |
| train        | 00000021 | gap >60s (max 2161s); track starts at t=2                           |
| train        | 00000022 | track starts at t=90                                                |
| train        | 00000024 | gap >60s (max 458s)                                                 |
| train        | 00000025 | track starts at t=2                                                 |
| train        | 00000026 | fragmented network (largest SCC 87%)                                |
| train        | 00000028 | gap >60s (max 196s)                                                 |
| train        | 00000032 | implausible jumps (max 300 km/h); track starts at t=2               |
| train        | 00000033 | implausible jumps (max 511 km/h); route breaks: 1                   |
| train        | 00000034 | track starts at t=2                                                 |
| train        | 00000035 | fragmented network (largest SCC 87%)                                |
| train        | 00000041 | gap >60s (max 89s); route breaks: 1                                 |
| train        | 00000042 | gap >60s (max 670s); route breaks: 1                                |
| train        | 00000043 | gap >60s (max 230s); implausible jumps (max 474 km/h)               |
| train        | 00000044 | gap >60s (max 341s)                                                 |
| train        | 00000046 | track starts at t=21                                                |
| train        | 00000053 | implausible jumps (max 256 km/h)                                    |
| train        | 00000054 | gap >60s (max 79s)                                                  |
| train        | 00000055 | gap >60s (max 184s)                                                 |
| train        | 00000057 | implausible jumps (max 302 km/h)                                    |
| train        | 00000060 | track starts at t=17                                                |
| train        | 00000062 | track starts at t=11                                                |
| train        | 00000063 | track starts at t=3                                                 |
| train        | 00000068 | track starts at t=13                                                |
| train        | 00000069 | gap >60s (max 2658s); route breaks: 1; route revisits arcs: 29      |
| train        | 00000070 | gap >60s (max 152s); track starts at t=2                            |
| train        | 00000072 | fragmented network (largest SCC 89%)                                |
| train        | 00000075 | route breaks: 2; route revisits arcs: 2                             |
| train        | 00000078 | gap >60s (max 52810s); track starts at t=3                          |
| train        | 00000079 | implausible jumps (max 648 km/h)                                    |
| train        | 00000080 | route breaks: 1                                                     |
| train        | 00000081 | gap >60s (max 536s); track starts at t=3                            |
| train        | 00000084 | gap >60s (max 312s); route revisits arcs: 2                         |
| train        | 00000085 | gap >60s (max 70s)                                                  |
| train        | 00000086 | route breaks: 1; route revisits arcs: 2                             |
| train        | 00000088 | gap >60s (max 232s)                                                 |
| train        | 00000089 | gap >60s (max 193s); route breaks: 1                                |
| train        | 00000090 | gap >60s (max 153064s); track starts at t=4; route revisits arcs: 1 |
| train        | 00000092 | implausible jumps (max 306 km/h); route revisits arcs: 14           |
| train        | 00000095 | route revisits arcs: 7; fragmented network (largest SCC 69%)        |

## 12. Reference route availability and structure

Trips with a route: 90 / 90

|                            |   mean |   min |    50% |     max |
|:---------------------------|-------:|------:|-------:|--------:|
| n_route_arcs               | 386.81 | 59    | 272.5  | 2266    |
| route_repeated_arcs        |   3.58 |  0    |   0    |  255    |
| route_breaks               |   0.16 |  0    |   0    |    3    |
| route_contiguous_pct       |  99.94 | 98.13 | 100    |  100    |
| route_start_to_gps_start_m |  61.78 |  1.34 |  33.22 |  457.27 |
| route_end_to_gps_end_m     |  60.34 |  1.11 |  27.12 |  548.14 |
| route_vs_gps_length_ratio  |   0.78 |  0.13 |   0.85 |    0.99 |
