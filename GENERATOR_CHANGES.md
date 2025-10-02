# Generator Working - Changes Summary

## Modifications Made

### 1. **Output Format: Pickle instead of CSV**
- Saves as compressed pickle file (`results_<timestamp>.pkl.gz`)
- Matches the format of `original_generator.py`
- Preserves all data without truncation

### 2. **Timestamp in Filename**
- Uses `time.time()` to add unique timestamp
- Format: `results_1728000000.pkl.gz`
- Prevents overwriting previous generated files

### 3. **Removed Images to Reduce File Size**
- `all_imgs` is now an empty list `[]`
- Only stores `all_obs` and `all_actions`
- Significantly reduces file size (images were the largest data)

### 4. **Minimum Trajectory Length Filter**
- Only keeps trajectories with **more than 5 steps**
- Skips short trajectories that don't provide enough information
- Prints warning when trajectory is too short

### 5. **Output Structure (matches original_generator.py)**
```python
{
    'size': int,
    'layout_id': int,
    'initial_distance': int,
    'scenario_id': int,
    'hidden_cost_type': int,
    'start_positions': tuple,    # Target starting position
    'start_directions': int,     # Target starting direction
    'goals': list,
    'goal': tuple,
    'all_actions': list,         # Action objects
    'all_imgs': [],              # Empty to save space
    'all_obs': list              # Observations
}
```

## Usage

```bash
cd /mnt/c/active_gr/neurosymbolic_agr/new_multi_grid
python3 multigrid/experience/generator_working.py
```

## Output Example

```
results_1728123456.pkl.gz  # With timestamp
```

## Benefits

1. ✅ **Smaller file size** - No images stored
2. ✅ **No overwriting** - Timestamp prevents conflicts
3. ✅ **Better quality** - Only meaningful trajectories (>5 steps)
4. ✅ **Compatible** - Same format as original_generator.py
5. ✅ **Complete data** - No CSV truncation issues
