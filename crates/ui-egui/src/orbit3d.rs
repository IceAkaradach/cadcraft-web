//! 3D Orbit & Modeling Interactive Viewer for CADCraft (Desktop and Web).
//!
//! Provides interactive 3D camera navigation (orbit rotation, panning, zooming),
//! preset views (Top, Front, Right, SW/SE/NE/NW Isometric), visual styles (Shaded,
//! Wireframe, X-Ray), depth-sorted rendering of 3D entities (Face3d, Line, Polyline3d,
//! Circle, Arc, Solid), and quick 3D primitive creation (Box, Cylinder, Pyramid).

use std::f32::consts::PI;
use egui::{Color32, Pos2, Rect, RichText, Shape, Stroke, vec2};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use cadcraft_color::Color;
use cadcraft_doc::EntityKind;
use cadcraft_geom::Vec3;
use crate::CadApp;

pub const ORBIT_STATE_ID: &str = "cadcraft_orbit3d_state";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Orbit3dState {
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub pan_x: f32,
    pub pan_y: f32,
    pub target_x: f64,
    pub target_y: f64,
    pub target_z: f64,
    pub visual_style: u8, // 0: Shaded with Edges, 1: Wireframe, 2: X-Ray
    pub show_grid: bool,
    pub show_axes: bool,
    pub initialized: bool,
}

impl Default for Orbit3dState {
    fn default() -> Self {
        Self {
            yaw: PI / 4.0,   // 45 degrees (SW Iso)
            pitch: 0.61548,  // ~35.26 degrees (Isometric tilt)
            zoom: 3.5,
            pan_x: 0.0,
            pan_y: 0.0,
            target_x: 0.0,
            target_y: 0.0,
            target_z: 0.0,
            visual_style: 0,
            show_grid: true,
            show_axes: true,
            initialized: false,
        }
    }
}

pub enum RenderElement {
    Line {
        p0: Pos2,
        p1: Pos2,
        stroke: Stroke,
        depth: f32,
    },
    Poly {
        pts: Vec<Pos2>,
        fill: Color32,
        stroke: Stroke,
        depth: f32,
    },
    Text {
        pos: Pos2,
        text: String,
        color: Color32,
        depth: f32,
    },
}

impl RenderElement {
    pub fn depth(&self) -> f32 {
        match self {
            RenderElement::Line { depth, .. } => *depth,
            RenderElement::Poly { depth, .. } => *depth,
            RenderElement::Text { depth, .. } => *depth,
        }
    }
}

pub struct Projector {
    center: Pos2,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    cos_yaw: f32,
    sin_yaw: f32,
    cos_pitch: f32,
    sin_pitch: f32,
    target: Vec3,
}

impl Projector {
    pub fn new(center: Pos2, st: &Orbit3dState) -> Self {
        Self {
            center,
            zoom: st.zoom,
            pan_x: st.pan_x,
            pan_y: st.pan_y,
            cos_yaw: st.yaw.cos(),
            sin_yaw: st.yaw.sin(),
            cos_pitch: st.pitch.cos(),
            sin_pitch: st.pitch.sin(),
            target: Vec3::new(st.target_x, st.target_y, st.target_z),
        }
    }

    pub fn project(&self, p: Vec3) -> (Pos2, f32) {
        let dx = (p.x - self.target.x) as f32;
        let dy = (p.y - self.target.y) as f32;
        let dz = (p.z - self.target.z) as f32;

        let x1 = dx * self.cos_yaw - dy * self.sin_yaw;
        let y1 = dx * self.sin_yaw + dy * self.cos_yaw;
        let z1 = dz;

        let u = x1;
        let v = -y1 * self.sin_pitch + z1 * self.cos_pitch;
        let depth = y1 * self.cos_pitch + z1 * self.sin_pitch;

        let sx = self.center.x + self.pan_x + u * self.zoom;
        let sy = self.center.y + self.pan_y - v * self.zoom;
        (Pos2::new(sx, sy), depth)
    }
}

/// Render the 3D scene (entities, ground grid, 3D UCS, lighting, and status indicator) onto any egui Painter.
pub fn render_3d_viewport(
    painter: &egui::Painter,
    canvas_rect: Rect,
    st: &mut Orbit3dState,
    doc: &cadcraft_doc::Drawing,
) {
    // 1. Dark CAD Viewport Background
    painter.rect_filled(canvas_rect, 0.0, Color32::from_rgb(0x1e, 0x22, 0x2b));

    // 2. Auto-center target if not initialized
    if !st.initialized {
        let mut min_p = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max_p = Vec3::new(f64::MIN, f64::MIN, f64::MIN);
        let mut count = 0usize;

        for entity in doc.model.iter() {
            match &entity.kind {
                EntityKind::Face3d(f) => {
                    for c in &f.corners {
                        min_p.x = min_p.x.min(c.x);
                        min_p.y = min_p.y.min(c.y);
                        min_p.z = min_p.z.min(c.z);
                        max_p.x = max_p.x.max(c.x);
                        max_p.y = max_p.y.max(c.y);
                        max_p.z = max_p.z.max(c.z);
                    }
                    count += 1;
                }
                EntityKind::Line(l) => {
                    for c in [&l.a, &l.b] {
                        min_p.x = min_p.x.min(c.x);
                        min_p.y = min_p.y.min(c.y);
                        min_p.z = min_p.z.min(c.z);
                        max_p.x = max_p.x.max(c.x);
                        max_p.y = max_p.y.max(c.y);
                        max_p.z = max_p.z.max(c.z);
                    }
                    count += 1;
                }
                EntityKind::Polyline3d(p) => {
                    for c in &p.points {
                        min_p.x = min_p.x.min(c.x);
                        min_p.y = min_p.y.min(c.y);
                        min_p.z = min_p.z.min(c.z);
                        max_p.x = max_p.x.max(c.x);
                        max_p.y = max_p.y.max(c.y);
                        max_p.z = max_p.z.max(c.z);
                    }
                    count += 1;
                }
                EntityKind::Circle(c) => {
                    min_p.x = min_p.x.min(c.center.x - c.radius);
                    min_p.y = min_p.y.min(c.center.y - c.radius);
                    min_p.z = min_p.z.min(c.center.z);
                    max_p.x = max_p.x.max(c.center.x + c.radius);
                    max_p.y = max_p.y.max(c.center.y + c.radius);
                    max_p.z = max_p.z.max(c.center.z);
                    count += 1;
                }
                _ => {}
            }
        }

        if count > 0 && min_p.x < max_p.x && min_p.y < max_p.y {
            st.target_x = (min_p.x + max_p.x) * 0.5;
            st.target_y = (min_p.y + max_p.y) * 0.5;
            st.target_z = (min_p.z + max_p.z) * 0.5;
            let span = (max_p.x - min_p.x).max(max_p.y - min_p.y).max(max_p.z - min_p.z);
            if span > 0.001 {
                st.zoom = (canvas_rect.height().min(canvas_rect.width()) * 0.5 / span as f32).clamp(0.5, 30.0);
            }
        }
        st.initialized = true;
    }

    // 3. Setup Projector & Collect Elements
    let proj = Projector::new(canvas_rect.center(), st);
    let mut elements: Vec<RenderElement> = Vec::new();

    // 3.1 Ground Grid (XY Plane)
    if st.show_grid {
        let grid_range = 100.0f64;
        let step = 20.0f64;
        let grid_stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(100, 120, 150, 45));

        let mut y = -grid_range;
        while y <= grid_range + 0.01 {
            let (p0, d0) = proj.project(Vec3::new(-grid_range, y, 0.0));
            let (p1, d1) = proj.project(Vec3::new(grid_range, y, 0.0));
            elements.push(RenderElement::Line {
                p0,
                p1,
                stroke: grid_stroke,
                depth: (d0 + d1) * 0.5,
            });
            y += step;
        }

        let mut x = -grid_range;
        while x <= grid_range + 0.01 {
            let (p0, d0) = proj.project(Vec3::new(x, -grid_range, 0.0));
            let (p1, d1) = proj.project(Vec3::new(x, grid_range, 0.0));
            elements.push(RenderElement::Line {
                p0,
                p1,
                stroke: grid_stroke,
                depth: (d0 + d1) * 0.5,
            });
            x += step;
        }
    }

    // 3.2 UCS 3D Coordinate Axes at Origin (0,0,0)
    if st.show_axes {
        let axis_len = 50.0f64;
        let (origin_pos, o_depth) = proj.project(Vec3::ZERO);

        // X Axis (Red)
        let (x_pos, x_depth) = proj.project(Vec3::new(axis_len, 0.0, 0.0));
        elements.push(RenderElement::Line {
            p0: origin_pos,
            p1: x_pos,
            stroke: Stroke::new(2.5, Color32::from_rgb(0xe0, 0x48, 0x48)),
            depth: (o_depth + x_depth) * 0.5,
        });
        elements.push(RenderElement::Text {
            pos: x_pos + vec2(4.0, -4.0),
            text: "X".to_string(),
            color: Color32::from_rgb(0xe0, 0x48, 0x48),
            depth: x_depth,
        });

        // Y Axis (Green)
        let (y_pos, y_depth) = proj.project(Vec3::new(0.0, axis_len, 0.0));
        elements.push(RenderElement::Line {
            p0: origin_pos,
            p1: y_pos,
            stroke: Stroke::new(2.5, Color32::from_rgb(0x48, 0xd0, 0x48)),
            depth: (o_depth + y_depth) * 0.5,
        });
        elements.push(RenderElement::Text {
            pos: y_pos + vec2(4.0, -4.0),
            text: "Y".to_string(),
            color: Color32::from_rgb(0x48, 0xd0, 0x48),
            depth: y_depth,
        });

        // Z Axis (Blue)
        let (z_pos, z_depth) = proj.project(Vec3::new(0.0, 0.0, axis_len));
        elements.push(RenderElement::Line {
            p0: origin_pos,
            p1: z_pos,
            stroke: Stroke::new(2.5, Color32::from_rgb(0x48, 0x90, 0xf0)),
            depth: (o_depth + z_depth) * 0.5,
        });
        elements.push(RenderElement::Text {
            pos: z_pos + vec2(4.0, -4.0),
            text: "Z".to_string(),
            color: Color32::from_rgb(0x48, 0x90, 0xf0),
            depth: z_depth,
        });
    }

    // 3.3 Document 3D Entities
    let light_dir = Vec3::new(-0.35, -0.55, 0.75).normalized();

    for entity in doc.model.iter() {
        let (base_r, base_g, base_b) = match entity.common.color {
            Color::True(rgb) => (rgb.0, rgb.1, rgb.2),
            _ => (110, 160, 220), // Default attractive CAD blue
        };

        match &entity.kind {
            EntityKind::Face3d(face) => {
                let is_tri = face.corners[2] == face.corners[3];
                let c0 = face.corners[0];
                let c1 = face.corners[1];
                let c2 = face.corners[2];
                let norm = (c1 - c0).cross(c2 - c0);
                let n_len = norm.len();
                let norm = if n_len > 1e-6 { norm * (1.0 / n_len) } else { Vec3::Z };
                let intensity = (0.35 + 0.65 * norm.dot(light_dir).abs()) as f32;

                let r = (base_r as f32 * intensity).clamp(0.0, 255.0) as u8;
                let g = (base_g as f32 * intensity).clamp(0.0, 255.0) as u8;
                let b = (base_b as f32 * intensity).clamp(0.0, 255.0) as u8;

                let (fill, stroke_col) = match st.visual_style {
                    1 => (Color32::TRANSPARENT, Color32::from_rgb(r, g, b)), // Wireframe
                    2 => (Color32::from_rgba_unmultiplied(r, g, b, 130), Color32::from_rgb(45, 60, 80)), // X-Ray
                    _ => (Color32::from_rgb(r, g, b), Color32::from_rgb(25, 30, 40)), // Shaded with edges
                };

                let (p0, d0) = proj.project(face.corners[0]);
                let (p1, d1) = proj.project(face.corners[1]);
                let (p2, d2) = proj.project(face.corners[2]);

                let (pts, avg_depth) = if is_tri {
                    (vec![p0, p1, p2], (d0 + d1 + d2) / 3.0)
                } else {
                    let (p3, d3) = proj.project(face.corners[3]);
                    (vec![p0, p1, p2, p3], (d0 + d1 + d2 + d3) * 0.25)
                };

                elements.push(RenderElement::Poly {
                    pts,
                    fill,
                    stroke: Stroke::new(1.0, stroke_col),
                    depth: avg_depth,
                });
            }
            EntityKind::Line(line) => {
                let (p0, d0) = proj.project(line.a);
                let (p1, d1) = proj.project(line.b);
                elements.push(RenderElement::Line {
                    p0,
                    p1,
                    stroke: Stroke::new(1.5, Color32::from_rgb(base_r, base_g, base_b)),
                    depth: (d0 + d1) * 0.5,
                });
            }
            EntityKind::Polyline3d(poly) => {
                if poly.points.len() >= 2 {
                    for i in 0..poly.points.len() - 1 {
                        let (p0, d0) = proj.project(poly.points[i]);
                        let (p1, d1) = proj.project(poly.points[i + 1]);
                        elements.push(RenderElement::Line {
                            p0,
                            p1,
                            stroke: Stroke::new(1.5, Color32::from_rgb(base_r, base_g, base_b)),
                            depth: (d0 + d1) * 0.5,
                        });
                    }
                    if poly.closed && poly.points.len() >= 3 {
                        let (p0, d0) = proj.project(poly.points[poly.points.len() - 1]);
                        let (p1, d1) = proj.project(poly.points[0]);
                        elements.push(RenderElement::Line {
                            p0,
                            p1,
                            stroke: Stroke::new(1.5, Color32::from_rgb(base_r, base_g, base_b)),
                            depth: (d0 + d1) * 0.5,
                        });
                    }
                }
            }
            EntityKind::Circle(c) => {
                let segs = 32usize;
                let mut prev_pt: Option<(Pos2, f32)> = None;
                let mut first_pt: Option<(Pos2, f32)> = None;
                for i in 0..segs {
                    let ang = (i as f64) * 2.0 * std::f64::consts::PI / (segs as f64);
                    let pt3 = c.center + Vec3::new(c.radius * ang.cos(), c.radius * ang.sin(), 0.0);
                    let projected = proj.project(pt3);
                    if let Some((prev_p, prev_d)) = prev_pt {
                        elements.push(RenderElement::Line {
                            p0: prev_p,
                            p1: projected.0,
                            stroke: Stroke::new(1.2, Color32::from_rgb(base_r, base_g, base_b)),
                            depth: (prev_d + projected.1) * 0.5,
                        });
                    } else {
                        first_pt = Some(projected);
                    }
                    prev_pt = Some(projected);
                }
                if let (Some(prev), Some(first)) = (prev_pt, first_pt) {
                    elements.push(RenderElement::Line {
                        p0: prev.0,
                        p1: first.0,
                        stroke: Stroke::new(1.2, Color32::from_rgb(base_r, base_g, base_b)),
                        depth: (prev.1 + first.1) * 0.5,
                    });
                }
            }
            EntityKind::Solid(solid) => {
                let (p0, d0) = proj.project(solid.corners[0]);
                let (p1, d1) = proj.project(solid.corners[1]);
                let (p2, d2) = proj.project(solid.corners[2]);
                let (p3, d3) = proj.project(solid.corners[3]);
                elements.push(RenderElement::Poly {
                    pts: vec![p0, p1, p3, p2],
                    fill: Color32::from_rgb(base_r, base_g, base_b),
                    stroke: Stroke::new(1.0, Color32::from_rgb(20, 25, 35)),
                    depth: (d0 + d1 + d2 + d3) * 0.25,
                });
            }
            _ => {}
        }
    }

    // 4. Painter's Algorithm: Sort from furthest to closest
    elements.sort_by(|a, b| b.depth().partial_cmp(&a.depth()).unwrap_or(std::cmp::Ordering::Equal));

    // 5. Draw Elements
    for el in elements {
        match el {
            RenderElement::Line { p0, p1, stroke, .. } => {
                if canvas_rect.contains(p0) || canvas_rect.contains(p1) {
                    painter.line_segment([p0, p1], stroke);
                }
            }
            RenderElement::Poly { pts, fill, stroke, .. } => {
                if pts.iter().any(|p| canvas_rect.contains(*p)) {
                    painter.add(Shape::convex_polygon(pts, fill, stroke));
                }
            }
            RenderElement::Text { pos, text, color, .. } => {
                if canvas_rect.contains(pos) {
                    painter.text(pos, egui::Align2::LEFT_TOP, text, egui::FontId::proportional(12.0), color);
                }
            }
        }
    }

    // 6. Status Indicator Overlay in Bottom-Left
    let view_style_name = match st.visual_style {
        1 => "Wireframe",
        2 => "X-Ray",
        _ => "Shaded",
    };
    let status_msg = format!("3D Orbit | Style: {} | Zoom: {:.1}x | Yaw: {:.0}° Pitch: {:.0}°",
        view_style_name, st.zoom, st.yaw.to_degrees(), st.pitch.to_degrees());
    painter.text(
        canvas_rect.left_bottom() + vec2(10.0, -20.0),
        egui::Align2::LEFT_BOTTOM,
        status_msg,
        egui::FontId::proportional(11.0),
        Color32::from_rgba_unmultiplied(160, 180, 205, 180),
    );
}

/// Show the 3D Orbit Viewer window.
pub fn dialog(app: &mut CadApp, ctx: &egui::Context, open: &mut bool) {
    let state_id = egui::Id::new(ORBIT_STATE_ID);
    let mut st: Orbit3dState = ctx.data_mut(|d| d.get_temp(state_id)).unwrap_or_default();
    let mut to_run: Option<(&'static str, Value)> = None;

    egui::Window::new("🌐 3D Orbit & Modeling")
        .id(egui::Id::new("3d_orbit_window"))
        .open(open)
        .default_size(vec2(780.0, 560.0))
        .min_size(vec2(480.0, 360.0))
        .resizable(true)
        .show(ctx, |ui| {
            // Top Toolbar: Preset Views & Visual Styles
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Views:").strong());
                if ui.button("Top").on_hover_text("Plan view from top (+Z)").clicked() {
                    st.yaw = 0.0;
                    st.pitch = PI / 2.0;
                }
                if ui.button("Front").on_hover_text("Elevation view from front (-Y)").clicked() {
                    st.yaw = 0.0;
                    st.pitch = 0.0;
                }
                if ui.button("Right").on_hover_text("Side view from right (+X)").clicked() {
                    st.yaw = -PI / 2.0;
                    st.pitch = 0.0;
                }
                if ui.button("SW Iso").on_hover_text("South-West Isometric view").clicked() {
                    st.yaw = PI / 4.0;
                    st.pitch = 0.61548;
                }
                if ui.button("SE Iso").on_hover_text("South-East Isometric view").clicked() {
                    st.yaw = -PI / 4.0;
                    st.pitch = 0.61548;
                }
                if ui.button("NE Iso").on_hover_text("North-East Isometric view").clicked() {
                    st.yaw = -3.0 * PI / 4.0;
                    st.pitch = 0.61548;
                }
                if ui.button("NW Iso").on_hover_text("North-West Isometric view").clicked() {
                    st.yaw = 3.0 * PI / 4.0;
                    st.pitch = 0.61548;
                }

                ui.separator();
                ui.label(RichText::new("Style:").strong());
                ui.selectable_value(&mut st.visual_style, 0, "Shaded");
                ui.selectable_value(&mut st.visual_style, 1, "Wireframe");
                ui.selectable_value(&mut st.visual_style, 2, "X-Ray");

                ui.separator();
                ui.checkbox(&mut st.show_grid, "Grid");
                ui.checkbox(&mut st.show_axes, "UCS");

                if ui.button("Reset View").on_hover_text("Recenter camera on 3D geometry").clicked() {
                    st.pan_x = 0.0;
                    st.pan_y = 0.0;
                    st.initialized = false;
                }
            });

            // 3D Creation Shortcuts Row
            ui.horizontal(|ui| {
                ui.label(RichText::new("Add 3D:").strong());
                if ui.button("+ 3D Box").on_hover_text("Create a 50×50×50 3D box solid").clicked() {
                    to_run = Some(("box", json!({ "length": 50.0, "width": 50.0, "height": 50.0 })));
                }
                if ui.button("+ 3D Cylinder").on_hover_text("Create a 3D cylinder (radius 25, height 50)").clicked() {
                    to_run = Some(("cylinder", json!({ "center": [70.0, 0.0, 0.0], "radius": 25.0, "height": 50.0, "sides": 16 })));
                }
                if ui.button("+ 3D Pyramid").on_hover_text("Create a 3D pyramid (radius 30, height 50)").clicked() {
                    to_run = Some(("pyramid", json!({ "center": [-70.0, 0.0, 0.0], "radius": 30.0, "height": 50.0, "sides": 4 })));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new("Left Drag: Orbit | Right/Shift Drag: Pan | Scroll: Zoom").small().weak());
                });
            });

            ui.separator();

            // 3D Canvas Area
            let avail = ui.available_size();
            let (response, painter) = ui.allocate_painter(avail, egui::Sense::drag());
            let canvas_rect = response.rect;

            // Handle Mouse Interactions inside Dialog
            if response.dragged() {
                let delta = response.drag_delta();
                let is_pan = ctx.input(|i| {
                    i.pointer.button_down(egui::PointerButton::Secondary) || i.modifiers.shift
                });
                if is_pan {
                    st.pan_x += delta.x;
                    st.pan_y += delta.y;
                } else if ctx.input(|i| i.pointer.button_down(egui::PointerButton::Primary)) {
                    st.yaw += delta.x * 0.01;
                    st.pitch = (st.pitch + delta.y * 0.01).clamp(-1.54, 1.54);
                }
            }

            if response.hovered() {
                let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                if scroll.abs() > 0.0 {
                    let factor = if scroll > 0.0 { 1.12 } else { 0.89 };
                    st.zoom = (st.zoom * factor).clamp(0.05, 500.0);
                }
            }

            // Render the 3D scene
            if let Ok(doc) = app.session.doc() {
                render_3d_viewport(&painter, canvas_rect, &mut st, &doc);
            }
        });

    // Save state back to egui temporary storage
    ctx.data_mut(|d| d.insert_temp(state_id, st));

    // Execute any requested 3D primitive commands outside of the borrow scope
    if let Some((cmd, val)) = to_run {
        let _ = app.run(cmd, val);
    }
}
