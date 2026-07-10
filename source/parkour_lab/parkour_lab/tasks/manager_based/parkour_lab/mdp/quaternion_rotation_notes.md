# Quaternion Rotation Notes

This file uses plain Markdown only. Mathematical expressions are written in code blocks or inline code so they render correctly even in Markdown viewers that do not support LaTeX or MathJax.

---

## Conventions Used Here

- Quaternion component order: `q = (w, x, y, z)`
- `w` is the scalar part.
- `(x, y, z)` is the vector part.
- Hamilton quaternion multiplication is used.
- The coordinate system is assumed to be right-handed.
- The rotation convention is the common active rotation:

```text
P_rotated = q * P * q^-1
```

Here, `P` is a 3D vector written as a pure quaternion.

---

## What Is a Quaternion?

A quaternion is a 4D number:

```text
q = w + x*i + y*j + z*k
```

It is often stored as four numbers:

```text
q = (w, x, y, z)
```

or as a scalar plus a 3D vector:

```text
q = (w, v)

where:
    w = scalar part
    v = (x, y, z) = vector part
```

For 3D rotations, we usually use unit quaternions. A unit quaternion has length 1:

```text
|q| = sqrt(w^2 + x^2 + y^2 + z^2) = 1
```

---

## Quaternion from Axis and Angle

A rotation by angle `theta` around a normalized axis

```text
u = (ux, uy, uz)
```

is represented by the unit quaternion:

```text
q = (
    cos(theta / 2),
    ux * sin(theta / 2),
    uy * sin(theta / 2),
    uz * sin(theta / 2)
)
```

The axis must be normalized:

```text
length(u) = sqrt(ux^2 + uy^2 + uz^2) = 1
```

Important: the quaternion stores half the rotation angle:

```text
theta_in_quaternion = theta / 2
```

---

## Example: 90 Degrees Around the Z Axis

Axis:

```text
u = (0, 0, 1)
```

Angle:

```text
theta = 90 degrees = pi / 2 radians
```

Half angle:

```text
theta / 2 = 45 degrees = pi / 4 radians
```

Quaternion:

```text
q = (
    cos(pi / 4),
    0,
    0,
    sin(pi / 4)
)
```

Since:

```text
cos(pi / 4) = sqrt(2) / 2
sin(pi / 4) = sqrt(2) / 2
```

we get:

```text
q = (sqrt(2) / 2, 0, 0, sqrt(2) / 2)
```

This rotation maps the x-axis to the y-axis:

```text
(1, 0, 0) -> (0, 1, 0)
```

---

## Rotating a 3D Vector

A 3D vector

```text
p = (px, py, pz)
```

is first written as a pure quaternion, meaning its scalar part is zero:

```text
P = (0, px, py, pz)
```

The rotated vector is computed as:

```text
P_rotated = q * P * q^-1
```

For unit quaternions, the inverse is just the conjugate:

```text
q^-1 = conjugate(q) = (w, -x, -y, -z)
```

The result has scalar part zero again:

```text
P_rotated = (0, px_rotated, py_rotated, pz_rotated)
```

The vector part is the rotated 3D vector:

```text
p_rotated = (px_rotated, py_rotated, pz_rotated)
```

---

## Quaternion Multiplication

Given two quaternions:

```text
q1 = (w1, v1)
q2 = (w2, v2)
```

where `v1` and `v2` are 3D vectors, their product is:

```text
q1 * q2 = (
    w1*w2 - dot(v1, v2),
    w1*v2 + w2*v1 + cross(v1, v2)
)
```

Component-wise, with

```text
q1 = (w1, x1, y1, z1)
q2 = (w2, x2, y2, z2)
```

the product is:

```text
q1 * q2 = (
    w1*w2 - x1*x2 - y1*y2 - z1*z2,
    w1*x2 + x1*w2 + y1*z2 - z1*y2,
    w1*y2 - x1*z2 + y1*w2 + z1*x2,
    w1*z2 + x1*y2 - y1*x2 + z1*w2
)
```

Quaternion multiplication is not commutative:

```text
q1 * q2 != q2 * q1
```

So the order of multiplication matters.

---

## Worked Example: Rotate `(1, 0, 0)` by 90 Degrees Around Z

Use:

```text
a = sqrt(2) / 2

q    = (a, 0, 0, a)
P    = (0, 1, 0, 0)
q^-1 = (a, 0, 0, -a)
```

First multiply `q * P`:

```text
q * P = (0, a, a, 0)
```

Then multiply by `q^-1`:

```text
(q * P) * q^-1 = (0, 0, 1, 0)
```

The scalar part is zero, so the rotated 3D vector is:

```text
p_rotated = (0, 1, 0)
```

So:

```text
(1, 0, 0) rotated 90 degrees around Z -> (0, 1, 0)
```

---

## Combining Rotations

If `q1` is applied first and `q2` is applied second, the combined rotation is:

```text
q_combined = q2 * q1
```

The first rotation appears on the right because the vector is transformed like this:

```text
P_rotated = q2 * (q1 * P * q1^-1) * q2^-1
```

This can be regrouped as:

```text
P_rotated = (q2 * q1) * P * (q2 * q1)^-1
```

Therefore:

```text
q_combined = q2 * q1
```

---

## Example: First Rotate Around Z, Then Around X

Start with:

```text
p = (1, 0, 0)
```

First rotate 90 degrees around Z:

```text
(1, 0, 0) -> (0, 1, 0)
```

Then rotate 90 degrees around X:

```text
(0, 1, 0) -> (0, 0, 1)
```

So the final result is:

```text
p_rotated = (0, 0, 1)
```

The corresponding quaternions are:

```text
q_z = (sqrt(2) / 2, 0, 0, sqrt(2) / 2)
q_x = (sqrt(2) / 2, sqrt(2) / 2, 0, 0)
```

Because `q_z` is applied first and `q_x` second, the combined quaternion is:

```text
q_combined = q_x * q_z
```

Not:

```text
q_combined = q_z * q_x
```

The order matters.

---

## Why `theta / 2`?

A rotation quaternion uses:

```text
q = (cos(theta / 2), axis * sin(theta / 2))
```

This is because the vector is multiplied from both sides:

```text
P_rotated = q * P * q^-1
```

Together, the left and right multiplication produce the full rotation angle `theta`.

---

## Why `q` and `-q` Represent the Same Rotation

The rotation formula is:

```text
P_rotated = q * P * q^-1
```

If we replace `q` with `-q`, we get:

```text
P_rotated = (-q) * P * (-q)^-1
```

The two minus signs cancel out, so the resulting rotation is the same.

Therefore:

```text
q and -q represent the same 3D rotation
```

---

## Python-Style Pseudocode

Create a quaternion from an axis and angle:

```python
axis = normalize(axis)

w = cos(theta / 2)
x = axis.x * sin(theta / 2)
y = axis.y * sin(theta / 2)
z = axis.z * sin(theta / 2)

q = (w, x, y, z)
```

Rotate a vector:

```python
P = (0, vector.x, vector.y, vector.z)

rotated = q * P * inverse(q)
```

For unit quaternions:

```python
inverse_q = (q.w, -q.x, -q.y, -q.z)
```

---

## Practical Notes

- Normalize rotation quaternions to avoid numerical drift.
- Quaternion multiplication order matters.
- `q` and `-q` represent the same rotation.
- Quaternions avoid gimbal lock.
- Quaternions are useful for smooth rotation interpolation.
- Spherical linear interpolation between two quaternions is called SLERP.

---

## Most Important Formulas

Quaternion from axis and angle:

```text
q = (
    cos(theta / 2),
    ux * sin(theta / 2),
    uy * sin(theta / 2),
    uz * sin(theta / 2)
)
```

Vector as pure quaternion:

```text
P = (0, px, py, pz)
```

Rotate vector:

```text
P_rotated = q * P * q^-1
```

Inverse of a unit quaternion:

```text
q^-1 = (w, -x, -y, -z)
```

Combine rotations:

```text
q_combined = q2 * q1
```

where `q1` is applied first and `q2` is applied second.
