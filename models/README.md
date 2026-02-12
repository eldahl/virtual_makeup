# Custom 3D Models for Head AR

Place your custom `.obj` files in this directory to use them with the AR application.

## OBJ File Requirements

- **Format**: Standard Wavefront OBJ format
- **Vertices**: Defined with `v x y z`
- **Faces**: Defined with `f v1 v2 v3` (triangles) or `f v1 v2 v3 v4` (quads)
- **Orientation**: 
  - Y-axis points UP (negative Y is above the head)
  - Model should be centered at origin
  - Scale: ~30-50 units radius works well

## Model Orientation Guide

```
     -Y (UP - above head)
      |
      |
      +---- +X (right)
     /
    /
   +Z (forward)
```

## Example Vertex Positions

- A hat brim at head level: `v 30 0 0`
- A hat peak above head: `v 0 -50 0`

## Tips

1. Keep models relatively simple (under 1000 vertices) for performance
2. Use centered geometry (origin at bottom center of model)
3. The model will automatically scale based on detected face size

## Testing Your Model

Modify `test.py` to load your custom model:

```python
# Add to MODELS dictionary:
MODELS['custom'] = {
    'create': lambda: load_obj('models/your_model.obj'),
    'color': (255, 128, 0),  # BGR color
    'name': 'My Custom Model'
}
```




