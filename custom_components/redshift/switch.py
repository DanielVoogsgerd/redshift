"""
Redshift for Home-Assistant.
Adapted to specify multiple setpoints

The idea was taken from https://github.com/KpaBap/hue-flux/
"""
from __future__ import annotations

import datetime
import logging
from typing import Any

import voluptuous as vol

from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_COLOR_TEMP,
    ATTR_RGB_COLOR,
    ATTR_TRANSITION,
    ATTR_XY_COLOR,
    DOMAIN as LIGHT_DOMAIN,
    VALID_TRANSITION,
    is_on,
)
from homeassistant.components.switch import DOMAIN, SwitchEntity
from homeassistant.const import (
    ATTR_ENTITY_ID,
    CONF_BRIGHTNESS,
    CONF_LIGHTS,
    CONF_MODE,
    CONF_NAME,
    CONF_PLATFORM,
    SERVICE_TURN_ON,
    STATE_ON,
    SUN_EVENT_SUNRISE,
    SUN_EVENT_SUNSET,
)
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv, event
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.sun import get_astral_event_date
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import slugify
from homeassistant.util.color import (
    color_RGB_to_xy_brightness,
    color_temperature_kelvin_to_mired,
    color_temperature_to_rgb,
)
from homeassistant.util.dt import as_local, utcnow as dt_utcnow

_LOGGER = logging.getLogger(__name__)

CONF_START_TIME = "start_time"
CONF_END_TIME = "end_time"
CONF_INTERVAL = "interval"
CONF_SETPOINTS = "setpoints"
CONF_TRANSITION = "transition"
CONF_SETPOINT_TIME = "time"
CONF_SETPOINT_CT = "temperature"
CONF_SETPOINT_BRIGHTNESS = "brightness"
CONF_SETPOINT_LIGHTS = "lights"

MODE_XY = "xy"
MODE_MIRED = "mired"
MODE_RGB = "rgb"
DEFAULT_MODE = MODE_XY

PLATFORM_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_PLATFORM): "redshift",
        vol.Optional(CONF_NAME, default="Redshift"): cv.string,
        vol.Required(CONF_START_TIME): cv.time,
        vol.Required(CONF_END_TIME): cv.time,
        vol.Required(CONF_LIGHTS): cv.entity_ids,
        vol.Required(CONF_INTERVAL, default=30): cv.positive_int,
        vol.Required(CONF_TRANSITION, default=30): VALID_TRANSITION,
        vol.Required(CONF_SETPOINTS): [{
            vol.Required(CONF_SETPOINT_TIME): cv.time,
            vol.Optional(CONF_SETPOINT_CT): vol.All(
                vol.Coerce(int), vol.Range(min=1000, max=40000)
            ),
            vol.Optional(CONF_SETPOINT_BRIGHTNESS): vol.All(
                vol.Coerce(int), vol.Range(min=0, max=255)
            ),
            vol.Optional(CONF_SETPOINT_LIGHTS): cv.entity_ids,
        }],
    }
)


async def async_set_lights_xy(hass, lights, x_val, y_val, brightness, transition):
    """Set color of array of lights."""
    for light in lights:
        if is_on(hass, light):
            service_data = {ATTR_ENTITY_ID: light}
            if x_val is not None and y_val is not None:
                service_data[ATTR_XY_COLOR] = [x_val, y_val]
            if brightness is not None:
                service_data[ATTR_BRIGHTNESS] = brightness
            if transition is not None:
                service_data[ATTR_TRANSITION] = transition
            await hass.services.async_call(LIGHT_DOMAIN, SERVICE_TURN_ON, service_data)


async def async_set_lights_temp(hass, lights, mired, brightness, transition):
    """Set color of array of lights."""
    for light in lights:
        if is_on(hass, light):
            service_data = {ATTR_ENTITY_ID: light}
            if mired is not None:
                service_data[ATTR_COLOR_TEMP] = int(mired)
            if brightness is not None:
                service_data[ATTR_BRIGHTNESS] = brightness
            if transition is not None:
                service_data[ATTR_TRANSITION] = transition
            await hass.services.async_call(LIGHT_DOMAIN, SERVICE_TURN_ON, service_data)


async def async_set_lights_rgb(hass, lights, rgb, transition):
    """Set color of array of lights."""
    for light in lights:
        if is_on(hass, light):
            service_data = {ATTR_ENTITY_ID: light}
            if rgb is not None:
                service_data[ATTR_RGB_COLOR] = rgb
            if transition is not None:
                service_data[ATTR_TRANSITION] = transition
            await hass.services.async_call(LIGHT_DOMAIN, SERVICE_TURN_ON, service_data)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:

    """Set up the Redshift switches."""
    name = config.get(CONF_NAME)
    lights = config.get(CONF_LIGHTS)
    start_time = config.get(CONF_START_TIME)
    end_time = config.get(CONF_END_TIME)
    interval = config.get(CONF_INTERVAL)
    transition = config.get(ATTR_TRANSITION)
    setpoints = config.get(CONF_SETPOINTS)
    redshift = RedshiftSwitch(
        name,
        hass,
        lights,
        start_time,
        end_time,
        interval,
        transition,
        setpoints,
    )
    async_add_entities([redshift])

    async def async_update(call: ServiceCall | None = None) -> None:
        """Update lights."""
        await redshift.async_redshift_update()

    service_name = slugify(f"{name} update")
    hass.services.async_register(DOMAIN, service_name, async_update)


class RedshiftSwitch(SwitchEntity, RestoreEntity):
    """Representation of a Redshift switch."""

    def __init__(
        self,
        name,
        hass,
        lights,
        start_time,
        end_time,
        interval,
        transition,
        setpoints,
    ):
        """Initialize the Redshift switch."""
        self._name = name
        self.hass = hass
        self._lights = lights
        self._start_time = start_time
        self._end_time = end_time
        self._interval = interval
        self._transition = transition
        self._setpoints = self.parse_setpoints(setpoints)
        self.unsub_tracker = None

    def parse_setpoints(self, config_setpoints):
        """Parse setpoints from config into a dictionary"""
        from collections import DefaultDict
        setpoints = DefaultDict(dict)
        for setpoint in config_setpoints:
            lights = setpoint["lights"] if setpoint["lights"] else self._lights
            time = setpoint["time"]
            brightness = setpoint["brightness"] if setpoint["brightness"] else setpoints[light][time][0]
            temperature = setpoint["temperature"] if setpoint["temperature"] else setpoints[light][time][1]
            for light in lights:
                setpoints[light][time] = (brightness, temperature)

        sorted_setpoints = {}
        for light, states in setpoints.items():
            sorted_setpoints[light] = list(sorted(states.items()))

        return sorted_setpoints

    @property
    def name(self):
        """Return the name of the device if any."""
        return self._name

    @property
    def is_on(self):
        """Return true if switch is on."""
        return self.unsub_tracker is not None

    async def async_added_to_hass(self) -> None:
        """Call when entity about to be added to hass."""
        last_state = await self.async_get_last_state()
        if last_state and last_state.state == STATE_ON:
            await self.async_turn_on()

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on redshift."""
        if self.is_on:
            return

        self.unsub_tracker = event.async_track_time_interval(
            self.hass,
            self.async_redshift_update,
            datetime.timedelta(seconds=self._interval),
        )

        # Make initial update
        await self.async_redshift_update()

        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off redshift."""
        if self.is_on:
            self.unsub_tracker()
            self.unsub_tracker = None

        self.async_write_ha_state()

    async def async_redshift_update(self, utcnow=None):
        """Update all lights using redshift."""
        if utcnow is None:
            utcnow = dt_utcnow()

        now = as_local(utcnow)

        if now < self._start_time or now > self._end_time:
            return

        for light, states in self._setpoints.items():
            brightness_start = None
            brightness_end = None
            brightness_start_time = None
            brightness_end_time = None
            temperature_start = None
            temperature_end = None
            temperature_start_time = None
            temperature_end_time = None

            # find setpoints before and after now
            for time, (brightness, temperature) in states:
                if time < now:
                    if brightness is not None:
                        brightness_start_time = time
                        brightness_start = brightness
                    if temperature is not None:
                        temperature_start_time = time
                        temperature_start = temperature
                if time > now:
                    if brightness_end is None:
                        brightness_end_time = time
                        brightness_end = brightness
                    if temperature_end is None:
                        temperature_end_time = time
                        temperature_end = temperature

            if brightness_start is None:
                brightness = brightness_end
            elif brightness_end is None:
                brightness = brightness_start
            else:
                progress = (now - brightness_start_time) / (brightness_end_time - brightness_start_time)
                brightness = brightness_start + progress * (brightness_end - brightness_start)

            if temperature_start is None:
                temperature = temperature_end
            elif temperature_end is None:
                temperature = temperature_start
            else:
                progress = (now - temperature_start_time) / (temperature_end_time - temperature_start_time)
                temperature = temperature_start + progress * (temperature_end - temperature_start)

            mired = color_temperature_kelvin_to_mired(temperature)
            await async_set_lights_temp(
                self.hass, light, mired, brightness, self._transition
            )
            _LOGGER.debug(
                "Light %s updated to mired:%s brightness:%s at %s",
                light,
                mired,
                brightness,
                now,
            )

    async def async_redshift_update_old(self, utcnow=None):
        """Update all the lights using redshift."""
        if utcnow is None:
            utcnow = dt_utcnow()

        now = as_local(utcnow)

        sunset = get_astral_event_date(self.hass, SUN_EVENT_SUNSET, now.date())
        start_time = self.find_start_time(now)
        end_time = self.find_end_time(now)

        if end_time <= start_time:
            # stop_time does not happen in the same day as start_time
            if start_time < now:
                # stop time is tomorrow
                end_time += datetime.timedelta(days=1)
        elif now < start_time:
            # stop_time was yesterday since the new start_time is not reached
            end_time -= datetime.timedelta(days=1)

        if start_time < now < sunset:
            # Daytime
            time_state = "day"
            temp_range = abs(self._start_colortemp - self._sunset_colortemp)
            day_length = int(sunset.timestamp() - start_time.timestamp())
            seconds_from_start = int(now.timestamp() - start_time.timestamp())
            percentage_complete = seconds_from_start / day_length
            temp_offset = temp_range * percentage_complete
            if self._start_colortemp > self._sunset_colortemp:
                temp = self._start_colortemp - temp_offset
            else:
                temp = self._start_colortemp + temp_offset
        else:
            # Night time
            time_state = "night"

            if now < end_time:
                if end_time < start_time and end_time.day == sunset.day:
                    # we need to use yesterday's sunset time
                    sunset_time = sunset - datetime.timedelta(days=1)
                else:
                    sunset_time = sunset

                night_length = int(end_time.timestamp() - sunset_time.timestamp())
                seconds_from_sunset = int(now.timestamp() - sunset_time.timestamp())
                percentage_complete = seconds_from_sunset / night_length
            else:
                percentage_complete = 1

            temp_range = abs(self._sunset_colortemp - self._stop_colortemp)
            temp_offset = temp_range * percentage_complete
            if self._sunset_colortemp > self._stop_colortemp:
                temp = self._sunset_colortemp - temp_offset
            else:
                temp = self._sunset_colortemp + temp_offset
        rgb = color_temperature_to_rgb(temp)
        x_val, y_val, b_val = color_RGB_to_xy_brightness(*rgb)
        brightness = self._brightness if self._brightness else b_val
        if self._disable_brightness_adjust:
            brightness = None
        if self._mode == MODE_XY:
            await async_set_lights_xy(
                self.hass, self._lights, x_val, y_val, brightness, self._transition
            )
            _LOGGER.debug(
                "Lights updated to x:%s y:%s brightness:%s, %s%% "
                "of %s cycle complete at %s",
                x_val,
                y_val,
                brightness,
                round(percentage_complete * 100),
                time_state,
                now,
            )
        elif self._mode == MODE_RGB:
            await async_set_lights_rgb(self.hass, self._lights, rgb, self._transition)
            _LOGGER.debug(
                "Lights updated to rgb:%s, %s%% of %s cycle complete at %s",
                rgb,
                round(percentage_complete * 100),
                time_state,
                now,
            )
        else:
            # Convert to mired and clamp to allowed values
            mired = color_temperature_kelvin_to_mired(temp)
            await async_set_lights_temp(
                self.hass, self._lights, mired, brightness, self._transition
            )
            _LOGGER.debug(
                "Lights updated to mired:%s brightness:%s, %s%% "
                "of %s cycle complete at %s",
                mired,
                brightness,
                round(percentage_complete * 100),
                time_state,
                now,
            )

    def find_start_time(self, now):
        """Return sunrise or start_time if given."""
        if self._start_time:
            sunrise = now.replace(
                hour=self._start_time.hour, minute=self._start_time.minute, second=0
            )
        else:
            sunrise = get_astral_event_date(self.hass, SUN_EVENT_SUNRISE, now.date())
        return sunrise

    def find_end_time(self, now):
        """Return dusk or stop_time if given."""
        if self._end_time:
            dusk = now.replace(
                hour=self._end_time.hour, minute=self._end_time.minute, second=0
            )
        else:
            dusk = get_astral_event_date(self.hass, "dusk", now.date())
        return dusk
