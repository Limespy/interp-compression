# Algorithm



## TOC <!-- omit in toc -->

- [Algorithm](#algorithm)


## Overview

Generals tructure is

Finding where to compress
1. Pick a spot in data
2. Calculate approximation function for that spot
3. Verify by comparing the approximations to real ones
    - If within tolerance, save the data
    - Else got to step 1

Steps for making as single complete comporession step

```mermaid
---
title: Overview
---

flowchart TB

previous([Previous])
search1[First search stage]
search2[Second search stage]
record[Record]
storage[(Storage)]
next([Next])

previous -..-> search1

search1 -. Option 1 .-> record
search1 -. Option 2 .-> search2

search2 --> record

record --> storage

record -..-> next

```
### Expanded version

```mermaid
---
title: Detailed overview
---

flowchart TB

previous([Previous])
search2[Second search stage]
record[Record]
storage[(Storage)]
next([Next])

subgraph search1[First search stage]
    search1_start([Start])
    search1_check[Check]
    search1_is_valid{Passed}
    search1_return([Return])

    search1_start --> search1_check
    search1_check --> search1_is_valid
    search1_is_valid -. True .-> search1_start
    search1_is_valid -. False .-> search1_return
end

previous -.Estimate.-> search1_start

subgraph search2[Second search stage]
    search2_start([Start])
    search2_check[Check]
    search2_new[New Interval]
    search2_done{Done}
    search2_return([Return])

    search2_start --> search2_check
    search2_check --> search2_new
    search2_new --> search2_done
    search2_done -. False .-> search2_start
    search2_done -. True .-> search2_return
end


search1_return -. Option 1:
               Datum & Index .-> record
search1_return -. Option 2:
               Datum & Interval .-> search2_start

search2_return -- Datum & Index --> record

record -- Datum --> storage

record -. Estimate .-> next

```


## First search stage

Each first search stage iteration

```mermaid
---
title: First search stage
---

flowchart TB


start([Start])
is_valid{Passed}
estimate[Make Estimate]
next([Next])
return[Return]
return_search2([Second stage])
return_record([Record])

start -- Estimate --> check
start -. Estimate .-> estimate

subgraph check[Check]
    direction TB
    check_start([Start])
    check_data[(Data)]
    check_sampler[Sampler]
    check_approx[Approximation]
    check_validation[Validation]
    check_return([Return])

    check_start -- Estimate --> check_sampler

    check_data <--> check_sampler

    check_sampler -- Sample --> check_approx
    check_sampler -- Sample --> check_validation
    check_approx -- Approximation --> check_validation
    check_validation -- Datum & Result --> check_return

end

check_return -- Datum & Result --> is_valid

is_valid -. True: Datum & Result .-> estimate
is_valid -. False: Datum & Result .-> return

estimate -. New estimate, Datum & Index .-> next

start -. Datum & Index .-> return

return -. Option 1:
       Previous Datum & Index .-> return_record
return -. Option 2:
       Datums & Interval .-> return_search2

```

## Second search stage

## Record



## Checking

```mermaid
---
title: Checking
---

flowchart TB

start([Start])
return([Return])

subgraph data[Data]
    data_xy[(x, y)]
    data_xydy[(x, y, y')]
    data_xydyddy[(x, y, y', y'')]
end

subgraph sampler[Sampler]
    sampler_all[All values]
    sampler_sqrt[Square root]
end

start -- Estimate --> sampler
data <--> sampler

subgraph approx[Approximation]
    direction TB
    approx_start([Start])
    approx_line_ends[Line:
                     endponts]
    approx_line_lsq[Line:
                    least squares]
    approx_cubic_ends[Cubic:
                    endpoints]
    approx_cubic_lsq[Cubic:
                    least squares]
    approx_return([Return])

    approx_start -. Option 1:
                   Sample .-> approx_line_ends --> approx_return
    approx_start -. Option 2:
                   Sample .-> approx_line_lsq --> approx_return
    approx_start -. Option 3:
                   Sample .-> approx_cubic_ends --> approx_return
    approx_start -. Option 4:
                   Sample .-> approx_cubic_lsq--> approx_return

end

sampler -- Sample --> approx

subgraph validation[Validation]
    direction TB
    validation_start([Start])
    validation_sequential[Sequential]
    validation_tolerance[Tolerance]
    validation_batch[Batch]
    validation_return([Return])

    validation_start -. Option 1:
                       Sample .-> validation_sequential
    validation_sequential <-- Tolerance --> validation_tolerance
    validation_tolerance <-- Tolerance --> validation_batch
    validation_start -. Option 2:
                       Sample .-> validation_batch

    validation_sequential -- Boolean --> validation_return
    validation_batch -- Excess --> validation_return
end

sampler -- Sample --> validation
approx -- Parameters --> validation
validation --> return
```

### Sampler


### Approximation

### Validation


#### Sequential

```mermaid
---
title: Validation
---
flowchart TB

start([Start])
tol[Tolerance]
return([Return])

start -- Sample --> loop


subgraph loop[Loop]
    direction TB
    loop_start[Start]
    approx[Approximated]
    residual[Residual]
    in_tol[Within tolerance?]

    loop_start -- Datum y --> residual
    loop_start -- Datum x --> approx
    residual -- Residual --> in_tol
    approx -- Approximated y --> residual
    in_tol -. True .-> loop_start
end
start -- Parameters --> approx
approx -. Option 1:
         Approximated y .-> tol
loop_start -. Option 2:
             Datum y .-> tol
tol -- Tolerance --> in_tol

loop -. True .-> return
in_tol -. False:
         False .-> return


```

#### Batch

```mermaid
---
title: Validation
---
flowchart TB

start([Start])
tol[Tolerance]
approx[Approximated]
residual[Residual]
excess[Excess]
return([Return])


start -- Sample y --> residual
start -- Parameters --> approx
start -- Sample x --> approx
approx -- Approximated y --> residual
approx -. Option 1:
         Approximated y .-> tol
start -. Option 2:
        Sample y .-> tol



residual -- Residual --> excess
tol -- Tolerance --> excess
excess --> return
```

#### Tolerance

```mermaid
---
title: Tolerance
---
flowchart TB

start([Start])
rtol[Relative]
comb[Combining]
return([Return])

start -- y --> rtol
rtol -- Relative --> comb
start -- Absolute --> comb
comb -- Tolerance --> return
```
