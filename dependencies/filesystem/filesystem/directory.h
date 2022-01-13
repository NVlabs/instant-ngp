/*
 * reference https://docs.microsoft.com/zh-cn/cpp/c-runtime-library/reference/findfirst-functions?f1url=https%3A%2F%2Fmsdn.microsoft.com%2Fquery%2Fdev15.query%3FappId%3DDev15IDEF1%26l%3DZH-CN%26k%3Dk(CORECRT_IO%2F_findfirst);k(_findfirst);k(DevLang-C%2B%2B);k(TargetOS-Windows)%26rd%3Dtrue&view=vs-2019
 * reference http://www.man7.org/linux/man-pages/man3/opendir.3.html
 *
 * Copyright (c) 2019 tangm421 <tangm421@outlook.com>
 *
 * All rights reserved. Use of this source code is governed by a
 * BSD-style license that can be found in the LICENSE file.
 */

#pragma once

#include "path.h"

#if defined(_WIN32)
#include <io.h>
#else
#include <sys/types.h>
#include <dirent.h>
#endif

NAMESPACE_BEGIN(filesystem)

class directory : public path
{
public:
    directory(const path& dir) : path(dir), m_dir(dir) {}

    class iterator
    {
    public:
        iterator() /* default ctor indicates the end iterator */
#if defined(_WIN32)
            : m_handle(-1) {}
#else
            : m_handle(NULL), m_data(NULL) {}
#endif

        iterator(const directory& dir) {
            m_dir = dir;
#if defined(_WIN32)
            std::string search_path(dir.make_absolute().str() + "/*.*");
            m_handle = _findfirst(search_path.c_str(), &m_data);
            if (is_valid_handler())
            {
                m_result = m_dir / m_data.name;
            }
            else /* an error occurs or  reaching the end */
            {
                /* do nothing */
            }
#else
            m_handle = opendir(dir.make_absolute().str().c_str());
            ++*this; /* here we do find the first directory entry like the begin iterator does */
#endif
        }
        ~iterator() {
            if (is_valid_handler())
            {
#if defined(_WIN32)
                _findclose(m_handle);
                m_handle = -1;
#else
                closedir(m_handle);
                m_handle = NULL;
                m_data = NULL;
#endif
            }
        }

        iterator& operator++() {
            if (is_valid_handler())
            {
#if defined(_WIN32)
                if (_findnext(m_handle, &m_data))
                {
                    if (ENOENT == errno) /* reaching the end */
                    {
                        m_result = path();
                    }
                    else /* an error occurs */
                    {
                        /* do nothing because the next call of this function will not do anything */
                    }
                }
                else
                {
                    m_result = m_dir / m_data.name;
                }
#else
                errno = 0;
                m_data = readdir(m_handle);
                if (0 != errno) /* an error occurs */
                {
                    /* do nothing because the next call of this function will not do anything */
                }
                if (!m_data) /* reaching the end */
                {
                    m_result = path();
                }
                else
                {
                    m_result = m_dir / m_data->d_name;
                }
#endif
            }
            return *this;
        }
        bool operator!=(const iterator& rhs) const {
            return !(*this == rhs);
        }
        bool operator==(const iterator& rhs) const {
            return **this == *rhs;
        }
        const path& operator*() const {
            return m_result;
        }
        const path* operator->() const {
            return &m_result;
        }

    protected:
        bool is_valid_handler() const {
#if defined(_WIN32)
            return -1 != m_handle;
#else
            return NULL != m_handle;
#endif
        }

    private:
        path m_dir;
        path m_result;
#if defined(_WIN32)
        intptr_t m_handle;
        _finddata_t m_data;
#else
        DIR* m_handle;
        struct dirent* m_data;
#endif
    };

    iterator begin() const { return iterator(*this); }
    iterator end() const { static iterator static_end; return static_end; };

private:
    path m_dir;
};


NAMESPACE_END(filesystem)
