% Author: zhou
% Created: 2025/12/14
% description: Synthetic GPR full wavefield based on time-domain convolution

clear, clc

r_bg = 1;
[freq, xr, w_rand] = random_freq_xr_wrand();
c0 = 3e8;
dt = (0.005+rand*0.015)*1e-9;
time_window = w_rand * 1e-9;
t = 0:dt:time_window;
Nt = numel(t);
r_tr = rand * 0.005 + 0.005;
trace = round(xr/r_tr);
x = linspace(0, xr, trace);
Nx = numel(x);

nw = Nt;
tmax_w = (nw-1)*dt;
tvec = linspace(-tmax_w/2, tmax_w/2, nw);
pf = (pi^2) * (freq^2);
ricker = (1 - 2*pf*(tvec.^2)) .* exp(-pf*(tvec.^2));

eps_min = 3;
d_max = (c0/(2*sqrt(eps_min))) * time_window/2;
depth_axis = t * c0 ./ (2 * sqrt(eps_min * 4));
z_min_layer = d_max * 0.1;
z_max_layer = d_max * 0.9;
layer_depth_range = z_max_layer - z_min_layer;

eps_base = 4.0;
max_layers_target = randi([3,8]);
min_thick = layer_depth_range / max_layers_target / 1.1;
max_thick = layer_depth_range / max_layers_target * 1.5;

max_overall_attempts = 200;
overall_attempt = 0;
successful = false;

while ~successful && overall_attempt < max_overall_attempts
    overall_attempt = overall_attempt + 1;
    
    layer_surfaces_tmp = nan(max_layers_target, Nx);
    layer_eps_list_tmp = nan(1, max_layers_target);
    layer_tan_delta_list_tmp = nan(1, max_layers_target);
    layer_tau_list_tmp = nan(1, max_layers_target);
    layer_alpha_list_tmp = nan(1, max_layers_target);
    layer_count_tmp = 0;
    prev_surface = ones(1, Nx) * z_min_layer;
    fail_flag = false;
    
    max_layers = max_layers_target;
    
    for L_try = 1:max_layers
        generate_layer = true;
        attempts = 0;
        thick = NaN;
        shift = zeros(1, Nx);
        
        while generate_layer && attempts < 50
            attempts = attempts + 1;
            layer_type = randi([1, 3]);
            
            switch layer_type
                case 1
                    max_amplitude = min_thick * 0.4;
                    A = [0.002, 0.0015, 0.0025] * (0.5 + rand());
                    fre = 2*pi/Nx .* ([0.3 + 0.3*rand(), 0.6 + 0.3*rand(), 1 + rand()]);
                    ph = 2*pi * rand(1,3);
                    x_idx = 1:Nx;
                    shift = A(1)*sin(fre(1)*x_idx + ph(1)) + ...
                            A(2)*cos(fre(2)*x_idx + ph(2)) + ...
                            A(3)*sin(fre(3)*x_idx + ph(3));
                    shift = shift * (max_amplitude / max(abs(shift)));
                    
                case 2
                    shift = zeros(1, Nx);
                    
                case 3
                    max_angle = 20;
                    angle = (2*rand()-1) * max_angle;
                    slope = tand(angle);
                    x_center = mean(x);
                    shift = slope * (x - x_center);
                    max_shift_magnitude = min_thick * 0.5;
                    current_max_shift = max(abs(shift));
                    if current_max_shift > max_shift_magnitude
                        shift = shift * (max_shift_magnitude / current_max_shift);
                    end
            end
            
            prev_max = max(prev_surface);
            current_shift_min = min(prev_surface + shift);
            thick_candidate = prev_max + min_thick - current_shift_min;
            
            if (thick_candidate >= min_thick) && (thick_candidate <= max_thick) && ...
               (max(prev_surface + thick_candidate + shift) <= z_max_layer)
                thick = thick_candidate;
                generate_layer = false;
            else
                if attempts >= 35
                    generate_layer = false;
                    thick = NaN;
                    fail_flag = true;
                end
            end
        end
        
        if fail_flag || isnan(thick)
            fail_flag = true;
            break;
        end
        
        new_surface = prev_surface + thick + shift;
        if max(new_surface) > z_max_layer
            fail_flag = true;
            break;
        end
        if any(new_surface <= prev_surface)
            fail_flag = true;
            break;
        end
        
        layer_count_tmp = layer_count_tmp + 1;
        layer_surfaces_tmp(layer_count_tmp, :) = new_surface;
        layer_eps_list_tmp(layer_count_tmp) = 3 + 12*rand();
        layer_tan_delta_list_tmp(layer_count_tmp) = 0.001 + 0.099*rand();
        layer_tau_list_tmp(layer_count_tmp) = (0.1 + 0.9*rand()) * 1e-9;
        layer_alpha_list_tmp(layer_count_tmp) = 0.1 + 0.4*rand();
        prev_surface = new_surface;
        
        if max(prev_surface) >= z_max_layer * 0.95
            break;
        end
    end
    
    if ~fail_flag && layer_count_tmp >= 1
        valid_indices = 1:layer_count_tmp;
        
        basic_check = all(~isnan(layer_surfaces_tmp(valid_indices, :)), 'all') && ...
                     all(max(layer_surfaces_tmp(valid_indices, :), [], 2) <= z_max_layer + 1e-12);
        
        if basic_check
            layer_surfaces = layer_surfaces_tmp(valid_indices, :);
            layer_eps_list = layer_eps_list_tmp(valid_indices);
            layer_tan_delta_list = layer_tan_delta_list_tmp(valid_indices);
            layer_tau_list = layer_tau_list_tmp(valid_indices);
            layer_alpha_list = layer_alpha_list_tmp(valid_indices);
            layer_count = layer_count_tmp;
            successful = true;
        else
            successful = false;
        end
    else
        successful = false;
    end
end

if ~successful
    error('Failed to generate complete geological model within %d attempts.', max_overall_attempts);
end

vel_layers = c0 ./ sqrt(layer_eps_list);

n_r = xr / 2 * sqrt(freq/100/1e6);
n_targets = round(n_r/2 + rand() * (n_r - n_r/2)) + 4;
z_min_target = d_max / 5;
z_max_target = 3 * d_max / 5;

min_horizontal_spacing = 0.4;
min_vertical_spacing = 0.1;
max_attempts = 1000;

tg.x0 = zeros(1, n_targets);
tg.z0 = zeros(1, n_targets);
tg.eps = zeros(1, n_targets);

target_count = 0;
attempt_count = 0;

while target_count < n_targets && attempt_count < max_attempts
    attempt_count = attempt_count + 1;
    
    candidate_x = rand() * (max(x) - 2/5*xr) + 1/5*xr;
    candidate_z = z_min_target + (z_max_target - z_min_target) * rand();
    
    valid_position = true;
    
    for k = 1:target_count
        horizontal_distance = abs(candidate_x - tg.x0(k));
        vertical_distance = abs(candidate_z - tg.z0(k));
        
        if horizontal_distance < min_horizontal_spacing || vertical_distance < min_vertical_spacing
            valid_position = false;
            break;
        end
    end
    
    if valid_position
        target_count = target_count + 1;
        tg.x0(target_count) = candidate_x;
        tg.z0(target_count) = candidate_z;
        tg.eps(target_count) = rand() * 15 + 5;
    end
end

if target_count < n_targets
    n_targets = target_count;
    tg.x0 = tg.x0(1:n_targets);
    tg.z0 = tg.z0(1:n_targets);
    tg.eps = tg.eps(1:n_targets);
end

target_layer = zeros(1, n_targets);
for k = 1:n_targets
    [~, ix_k] = min(abs(x - tg.x0(k)));
    for L = 1:layer_count
        if L == 1
            if tg.z0(k) <= layer_surfaces(1, ix_k)
                target_layer(k) = 1;
                break;
            end
        else
            if layer_surfaces(L-1, ix_k) < tg.z0(k) && tg.z0(k) <= layer_surfaces(L, ix_k)
                target_layer(k) = L;
                break;
            end
        end
        if L == layer_count && tg.z0(k) > layer_surfaces(L, ix_k)
            target_layer(k) = layer_count + 1;
        end
    end
end

RC_diff = zeros(Nx, Nt);

max_angle = 0;
side_attenuation_factor = 1;

for ix = 1:Nx
    for k = 1:n_targets
        L = target_layer(k);
        if L == 1
            eps_layer = eps_base;
            tan_delta_layer = 0.001;
            tau_layer = 0.5e-9;
            alpha_layer = 0.3;
        elseif L <= layer_count
            eps_layer = layer_eps_list(L);
            tan_delta_layer = layer_tan_delta_list(L);
            tau_layer = layer_tau_list(L);
            alpha_layer = layer_alpha_list(L);
        else
            eps_layer = layer_eps_list(end);
            tan_delta_layer = layer_tan_delta_list(end);
            tau_layer = layer_tau_list(end);
            alpha_layer = layer_alpha_list(end);
        end
        
        rc_target = abs((sqrt(tg.eps(k)) - sqrt(eps_layer)) / (sqrt(tg.eps(k)) + sqrt(eps_layer)));
        
        if L <= layer_count
            v = vel_layers(L);
        else
            v = vel_layers(end);
        end
        dx = x(ix) - tg.x0(k);
        r = sqrt(dx^2 + tg.z0(k)^2);
        
        theta = atan2(abs(dx), tg.z0(k)) * 180/pi;
        
        if theta <= max_angle
            aa = cosd(theta)^2;
        else
            aa = cosd(theta)^4 * side_attenuation_factor;
        end
        
        ga = 1 / (1 + r^2);
        
        alpha_ohm = (pi * freq * sqrt(eps_layer) * tan_delta_layer) / c0;
        oa = exp(-alpha_ohm * r);

        omega = 2 * pi * freq;
        j = 1i;
        eps_inf = eps_layer * 0.8;
        epsilon_complex = eps_inf + (eps_layer - eps_inf) ./ (1 + (j * omega * tau_layer).^(1 - alpha_layer));
        
        epsilon_real = real(epsilon_complex);
        epsilon_imag = imag(epsilon_complex);
        
        if epsilon_real > 0
            alpha_pol = (2 * pi * freq * epsilon_imag) / (c0 * sqrt(epsilon_real));
            pa = exp(-alpha_pol * r);
        else
            pa = 1;
        end
        
        ta = aa * ga * oa * pa;
        
        tau = 2 * r / v;
        idx = round(tau / dt) + 1;
        
        if idx >= 1 && idx <= Nt
            RC_diff(ix, idx) = RC_diff(ix, idx) + rc_target * ta;
        end
    end
end

data_diff = zeros(Nt, Nx);
for i = 1:Nx
    trace_tmp = conv(RC_diff(i, :), ricker, 'same');
    data_diff(:, i) = trace_tmp(:);
end
data_diff = data_diff * 50;

% mask_diff = conv2(double(RC_diff' ~= 0), ones(nw, 1), 'same') > 0; % Nt x Nx logical
% for col = 1:Nx
%     bg_idx = ~mask_diff(:, col); % 背景位置
%     if any(bg_idx)
%         bg_mean = mean(data_diff(bg_idx, col));
%     else
%         bg_mean = mean(data_diff(:, col)); % 兜底
%     end
%     data_diff(:, col) = data_diff(:, col) - bg_mean;
% end
% data_diff = data_diff - mean(data_diff(:));
reflectivity = zeros(Nt, Nx);
for L = 1:layer_count
    if L == 1
        prev_eps = eps_base;
    else
        prev_eps = layer_eps_list(L-1);
    end
    curr_eps = layer_eps_list(L);
    R = (sqrt(curr_eps) - sqrt(prev_eps)) / (sqrt(curr_eps) + sqrt(prev_eps));
    for ix = 1:Nx
        tau = 0;
        for k2 = 1:L
            if k2 == 1
                d = layer_surfaces(1, ix);
            else
                d = layer_surfaces(k2, ix) - layer_surfaces(k2-1, ix);
            end
            tau = tau + 2 * d / vel_layers(k2);
        end
        t_idx = round(tau / dt) + 1;
        if t_idx >= 1 && t_idx <= Nt
            reflectivity(t_idx, ix) = reflectivity(t_idx, ix) + R;
        end
    end
end

reflect_wave = conv2(reflectivity, ricker', 'same') * 50;

fan_refl = 30;
max_abs_refl = max(reflect_wave(:));
if max_abs_refl > 0
    reflect_wave = reflect_wave / max_abs_refl * fan_refl;
else
    reflect_wave = zeros(size(reflect_wave));
end

fan_diff = 10 + rand *25;
max_abs_diff = max((data_diff(:)));
if max_abs_diff > 0
    data_diff = data_diff / max_abs_diff * fan_diff;
else
    data_diff = zeros(size(data_diff));
end
% mask_reflect = conv2(double(reflectivity~=0), ones(nw,1), 'same') > 0; % Nt x Nx logical
% for col = 1:Nx
%     bg_idx = ~mask_reflect(:, col);
%     if any(bg_idx)
%         bg_mean = mean(reflect_wave(bg_idx, col));
%     else
%         bg_mean = mean(reflect_wave(:, col));
%     end
%     reflect_wave(:, col) = reflect_wave(:, col) - bg_mean;
% end
% reflect_wave = reflect_wave - mean(reflect_wave(:));
r_diff_clipped = rand * 0.5 + 1.0;
r_refl_clipped = rand * 0.7 + 0.8;

data_diff = min(max(data_diff, min(data_diff(:))),min(data_diff(:))*(-r_diff_clipped));
reflect_wave = min(max(reflect_wave, min(reflect_wave(:))),min(reflect_wave(:))*(-r_refl_clipped));
cd = max(fan_diff(:));

[X, Z] = meshgrid(x, depth_axis);

fractal_background = zeros(Nt, Nx);
num_octaves = 2;
persistence = 100;

for octave = 1:num_octaves
    frequency = 2^(octave-1);
    amplitude = persistence^(octave-1);
    [noise_x, noise_z] = meshgrid(linspace(0, frequency, Nx), linspace(0, frequency, Nt));
    
    [grad_x, grad_z] = gradient(randn(size(noise_x)));
    octave_noise = cos(2*pi*noise_x) .* grad_x + sin(2*pi*noise_z) .* grad_z;
    octave_noise = (octave_noise - min(octave_noise(:))) / (max(octave_noise(:)) - min(octave_noise(:)));
    
    fractal_background = fractal_background + amplitude * octave_noise;
end

anomaly_field = zeros(Nt, Nx);
num_anomalies = 10 + randi(15);

for i = 1:num_anomalies
    center_x = rand() * Nx;
    center_z = rand() * Nt * 0.8;
    radius_x = 2 + rand() * 8;
    radius_z = 1 + rand() * 4;

    anomaly_mask = ((X - center_x).^2 / radius_x^2 + (Z - center_z).^2 / radius_z^2) <= 1;
    anomaly_strength = 0.1 + 5.4 * rand();
    anomaly_field(anomaly_mask) = anomaly_field(anomaly_mask) + anomaly_strength;
end

r_eps = rand;
background_eps = r_eps * fractal_background + (1 - r_eps) * anomaly_field;
background_eps = background_eps + 50 * randn(Nt, Nx);
background_eps = 0.1 + 4.9 * (background_eps - min(background_eps(:))) / (max(background_eps(:)) - min(background_eps(:)));
smooth_kernel = fspecial('gaussian', [7, 7], 1.2);
background_eps = imfilter(background_eps, smooth_kernel, 'replicate');

background_reflectivity = zeros(Nt, Nx);
base_eps = 4.0;

for i = 2:Nt
    for j = 1:Nx
        eps_prev = base_eps * (1 + 0.3 * background_eps(i-1, j));
        eps_curr = base_eps * (1 + 0.3 * background_eps(i, j));
        R_bg = (sqrt(eps_curr) - sqrt(eps_prev)) / (sqrt(eps_curr) + sqrt(eps_prev));
        background_reflectivity(i, j) = R_bg * 0.3;
    end
end

background_wavefield = conv2(background_reflectivity, ricker', 'same');
max_bg = max(abs(background_wavefield(:)));
background_wavefield = background_wavefield / max_bg * cd * (rand * 0.3 + 0.7);

full_wavefield = data_diff + reflect_wave;

noise_level = rand * 0.1;
noise = noise_level * max(abs(full_wavefield(:))) * randn(size(full_wavefield));
if r_bg == 1
    full_wavefield = full_wavefield + background_wavefield;
elseif r_bg == 2
    full_wavefield = full_wavefield + noise;
end
