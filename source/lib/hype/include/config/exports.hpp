/***********************************************************************************************************
Copyright (c) 2013, Robin Haberkorn, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/

#ifndef __HYPE_CONFIG_EXPORTS_HPP
#define __HYPE_CONFIG_EXPORTS_HPP

#ifdef _WIN32

#ifdef HYPE_MAKE_SHARED
#define HYPE_EXPORT __declspec(dllexport)
#elif defined(HYPE_USE_SHARED)
#define HYPE_EXPORT __declspec(dllimport)
#endif

#endif

#ifndef HYPE_EXPORT
#define HYPE_EXPORT
#endif

#endif
