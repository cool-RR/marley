// Copyright 2020 Ram Rachum and collaborators.
// This program is distributed under the MIT license.



function re_match_all(pattern, string) {
  regexp = new RegExp(pattern, 'g');
  array = [...string.matchAll(regexp)];
  return array
}


function animate() {
  if (window._animate_lock) return;
  window._animate_lock = true;
  try {
    animate_core()
  }
  finally {
    window._animate_lock = false;
    requestAnimationFrame(animate);
  }
}

_needle_on_screen = null;
ceil_needle = 0;
tau = Math.PI * 2;

BORDER_WIDTH = 2

player_avatars = new Map();


function update_ui() {
  document.getElementById('timepoint-text').textContent =
                                                  Math.abs(ceil_needle).toString().padStart(6, '0');
  timepoint_slider = document.getElementById('timepoint-slider');
  timepoint_slider.value = ceil_needle;
  timepoint_slider.max = __BRYTHON__.$getattr(window.timeline, '_transitions').length - 1;

  speed_slider = document.getElementById('speed-slider');
  speed_slider.value = __BRYTHON__.$getattr(window.timeline, 'speed');

}

function mix_colors(old_color, new_color, linear_transition_complement, linear_transition) {
  result = (
    65536 * Math.floor(old_color[0] * linear_transition_complement +
                       new_color[0] * linear_transition) +
    256 * Math.floor(old_color[1] * linear_transition_complement +
                     new_color[1] * linear_transition) +
    Math.floor(old_color[2] * linear_transition_complement + new_color[2] * linear_transition)
  );
  return '#' + result.toString(16).padStart(6, '0');
}

function draw_player(context, letter, x, y, color, opacity) {
    context.save();
    context.globalAlpha = opacity;
    context.translate(BORDER_WIDTH + x * CELL_SIZE + HALF_CELL_SIZE,
                      BORDER_WIDTH + y * CELL_SIZE + HALF_CELL_SIZE);

    context.fillStyle = color;
    context.beginPath();
    context.arc(0, 0, HALF_CELL_SIZE - 2, 0, tau);
    context.fill();

    context.font = player_font;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillStyle = '#cccccc';
    context.fillText(letter, player_text_x_shift, player_text_y_shift);
    context.restore();
}

function draw_food(context, x, y, opacity) {
    context.save();
    context.globalAlpha = opacity;
    context.translate(BORDER_WIDTH + x * CELL_SIZE + HALF_CELL_SIZE,
                      BORDER_WIDTH + y * CELL_SIZE + HALF_CELL_SIZE);

    context.fillStyle = '#666666';
    context.beginPath();
    context.arc(0, 0, HALF_CELL_SIZE / 6, 0, tau);
    context.fill();

    context.restore();
}


function draw_bullet(context, x, y, angle_offset, opacity) {
    context.save();
    context.globalAlpha = opacity;
    context.translate(BORDER_WIDTH + x * CELL_SIZE + HALF_CELL_SIZE,
                      BORDER_WIDTH + y * CELL_SIZE + HALF_CELL_SIZE);
    context.rotate(angle_offset);

    context.lineWidth = 0.2 * HALF_CELL_SIZE;

    context.lineCap = 'round';
    context.strokeStyle = '#666666';
    context.beginPath();
    context.moveTo(0, -0.45 * HALF_CELL_SIZE);
    context.lineTo(0, 0.33 * HALF_CELL_SIZE);
    context.stroke();

    //context.strokeStyle = '#444444';
    context.lineCap = 'butt';
    context.globalAlpha = opacity * 0.3;
    wind_factor = (1 + Math.sin(needle_modulo * tau));
    for (let direction = -1; direction < 2 ; direction += 2) {
      context.beginPath();
      context.moveTo(0.13 * HALF_CELL_SIZE * direction, -0.13 * HALF_CELL_SIZE);
      context.lineTo(wind_factor * 0.13 * HALF_CELL_SIZE * direction, 0.66 * HALF_CELL_SIZE);
      context.stroke();
    }

    context.restore();
}

function draw_wall(context, x, y, opacity) {
    context.save();
    context.globalAlpha = opacity;
    context.translate(BORDER_WIDTH + x * CELL_SIZE + HALF_CELL_SIZE,
                      BORDER_WIDTH + y * CELL_SIZE + HALF_CELL_SIZE);

    context.strokeStyle = '#aaaaaa';
    context.lineWidth = 8;
    context.strokeRect(- HALF_CELL_SIZE + 5, - HALF_CELL_SIZE + 5,
                       CELL_SIZE - 10, CELL_SIZE - 10);

    context.restore();
}


function draw_edge(context) {
    context.save();

    context.strokeStyle = '#aaaaaa';
    context.lineWidth = BORDER_WIDTH;
    context.strokeRect(0, 0, 600, 600);

    context.restore();
}



function get_target_and_whether_needle_reached_it() {
  if (_anchor_speed >= 0) {
    let target = transitions.length - 1;
    if (_anchor_target != false && _anchor_target < target) {
      target = _anchor_target;
    }
    return [target, (needle >= target)];
  } else {
    return [_anchor_target, (needle <= _anchor_target)];
  }
}

function animate_core() {
  time = Date.now() / 1_000;
  timeline = __BRYTHON__.$getattr(window, 'timeline');
  is_playing = __BRYTHON__.$getattr(timeline, 'is_playing');
  transitions = __BRYTHON__.$getattr(timeline, '_transitions');
  if (!transitions.length) return;
  if (is_playing) {
    _anchor_time = __BRYTHON__.$getattr(timeline, '_anchor_time');
    _anchor_needle = __BRYTHON__.$getattr(timeline, '_anchor_needle');
    _anchor_speed = __BRYTHON__.$getattr(timeline, '_anchor_speed');
    _anchor_target = __BRYTHON__.$getattr(timeline, '_anchor_target');
    _anchor_play_after_target = __BRYTHON__.$getattr(timeline, '_anchor_play_after_target');
    needle = _anchor_needle + (time - _anchor_time) * _anchor_speed;
    [target, needle_reached_target] = get_target_and_whether_needle_reached_it();
    if (needle_reached_target) {
      needle = target;
      if (_anchor_play_after_target) {
        __BRYTHON__.$getattr(window.timeline, '_set_anchor')();
      } else {
        __BRYTHON__.$getattr(window.timeline, 'pause')();
      }
    }
    __BRYTHON__.$setattr(timeline, '_needle', needle);
  }
  else {
    needle = __BRYTHON__.$getattr(timeline, '_needle');
  }
  if (_needle_on_screen == needle) return;
  ceil_needle = Math.ceil(needle);
  needle_modulo = needle % 1;

  if (ceil_needle != Math.ceil(_needle_on_screen)) update_ui();

  transition = __BRYTHON__.$getitem(transitions, ceil_needle);
  linear_transition_complement = ceil_needle - needle;
  linear_transition = 1 - linear_transition_complement;
  cute_transition_complement = Math.atan((-1 / linear_transition_complement) +
                                         (1 / (1 - linear_transition_complement))) / Math.PI + 0.5;
  cute_transition = 1 - cute_transition_complement;
  if (linear_transition <= 0.5) {
    eaten_food_transition = 0;
  } else if (linear_transition >= 0.7) {
    eaten_food_transition = 1;
  } else {
    eaten_food_transition = (linear_transition - 0.5) / 0.2
  }
  eaten_food_transition_complement = 1 - eaten_food_transition;

  canvas = document.getElementById('canvas');
  context = canvas.getContext('2d');
  context.clearRect(0, 0, 720, 720);

  BOARD_SIZE = __BRYTHON__.$getitem(transition, 'board_size');
  CELL_SIZE = (600 - 2 * BORDER_WIDTH) / BOARD_SIZE;
  HALF_CELL_SIZE = CELL_SIZE / 2;
  BOARD_WIDTH = BOARD_HEIGHT = CELL_SIZE * BOARD_SIZE;
  font_size = Math.ceil(CELL_SIZE * (2 / 3));
  player_font = ('bold ' + font_size.toString() +
                 'px Consolas, SFMono-Regular, "Liberation Mono", Menlo, monospace');
  player_text_x_shift = 0;
  player_text_y_shift = font_size / 14;

  draw_edge(context);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  //// Drawing players: ////////////////////////////////////////////////////////////////////////////
  //                                                                                              //
  players = __BRYTHON__.$getitem(transition, 'players');
  for (let i = 0; i < players.length; i++) {
    player = players[i];
    letter = player[0];
    old_position_x = player[1][0][0];
    new_position_x = player[1][1][0];
    old_position_y = player[1][0][1];
    new_position_y = player[1][1][1];
    old_color = player[2][0];
    new_color = player[2][1];
    old_opacity = player[3][0];
    new_opacity = player[3][1];

    position_x = cute_transition_complement * old_position_x + cute_transition * new_position_x;
    position_y = cute_transition_complement * old_position_y + cute_transition * new_position_y;

    color = mix_colors(old_color, new_color, linear_transition_complement, linear_transition);
    opacity = linear_transition_complement * old_opacity + linear_transition * new_opacity;

    draw_player(context, letter, position_x, position_y, color, opacity);
  }
  //                                                                                              //
  //// Finished drawing players. ///////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////////////
  //// Drawing bullets: ////////////////////////////////////////////////////////////////////////////
  //                                                                                              //
  bullets = __BRYTHON__.$getitem(transition, 'bullets');
  for (let i = 0; i < bullets.length; i++) {
    [[[old_bullet_x, old_bullet_y], [new_bullet_x, new_bullet_y]],
     angle_offset, [old_opacity, new_opacity]] = bullets[i];

    bullet_x = linear_transition_complement * old_bullet_x + linear_transition * new_bullet_x;
    bullet_y = linear_transition_complement * old_bullet_y + linear_transition * new_bullet_y;
    opacity = linear_transition_complement * old_opacity + linear_transition * new_opacity;

    draw_bullet(context, bullet_x, bullet_y, angle_offset, opacity);
  }
  //                                                                                              //
  //// Finished drawing bullets. ///////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////////////
  //// Drawing walls: //////////////////////////////////////////////////////////////////////////////
  //                                                                                              //
  walls = __BRYTHON__.$getitem(transition, 'walls');
  for (let i = 0; i < walls.length; i++) {
    [[wall_x, wall_y], [old_opacity, new_opacity]] = walls[i];
    opacity = linear_transition_complement * old_opacity + linear_transition * new_opacity;
    draw_wall(context, wall_x, wall_y, opacity);
  }
  //                                                                                              //
  //// Finished drawing walls. /////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////////////
  //// Drawing food: ///////////////////////////////////////////////////////////////////////////////
  //                                                                                              //
  food = __BRYTHON__.$getitem(transition, 'food');
  for (let i = 0; i < food.length; i++) {
    [[food_x, food_y], [old_opacity, new_opacity]] = food[i];
    if (old_opacity == 1 & new_opacity == 0) { // Food eaten
      opacity = (eaten_food_transition_complement * old_opacity +
                 eaten_food_transition * new_opacity);
    } else {
      opacity = linear_transition_complement * old_opacity + linear_transition * new_opacity;
    }

    draw_food(context, food_x, food_y, opacity);
  }
  //                                                                                              //
  //// Finished drawing food. //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////

  _needle_on_screen = needle;
}