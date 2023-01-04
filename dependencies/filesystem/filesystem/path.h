/*
	path.h -- A simple class for manipulating paths on Linux/Windows/Mac OS

	Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

	All rights reserved. Use of this source code is governed by a
	BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "fwd.h"
#include <algorithm>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cctype>
#include <cstdlib>
#include <cerrno>
#include <cstring>

#if defined(_WIN32)
# include <windows.h>
# include <ShlObj.h>
#else
# include <unistd.h>
#endif
#include <sys/stat.h>

#if defined(__linux)
# include <linux/limits.h>
#endif

NAMESPACE_BEGIN(filesystem)

/**
 * \brief Simple class for manipulating paths on Linux/Windows/Mac OS
 *
 * This class is just a temporary workaround to avoid the heavy boost
 * dependency until boost::filesystem is integrated into the standard template
 * library at some point in the future.
 */
class path {
public:
	enum path_type {
		windows_path = 0,
		posix_path = 1,
#if defined(_WIN32)
		native_path = windows_path
#else
		native_path = posix_path
#endif
	};

	path() : m_type(native_path), m_absolute(false), m_smb(false) { }

	path(const path &path)
		: m_type(path.m_type), m_path(path.m_path), m_absolute(path.m_absolute), m_smb(path.m_smb) {}

	path(path &&path)
		: m_type(path.m_type), m_path(std::move(path.m_path)),
		  m_absolute(path.m_absolute), m_smb(path.m_smb) {}

	path(const char *string) : m_smb(false) { set(string); }

	path(const std::string &string) : m_smb(false) { set(string); }

#if defined(_WIN32)
	path(const std::wstring &wstring) { set(wstring); }
	path(const wchar_t *wstring) { set(wstring); }
#endif

	size_t length() const { return m_path.size(); }

	bool empty() const { return m_path.empty(); }

	bool is_absolute() const { return m_absolute; }

	path make_absolute() const {
#if !defined(_WIN32)
		char temp[PATH_MAX];
		if (realpath(str().c_str(), temp) == NULL)
			throw std::runtime_error("Internal error in realpath(): " + std::string(strerror(errno)));
		return path(temp);
#else
		std::wstring value = wstr(), out(MAX_PATH_WINDOWS, '\0');
		DWORD length = GetFullPathNameW(value.c_str(), MAX_PATH_WINDOWS, &out[0], NULL);
		if (length == 0)
			throw std::runtime_error("Internal error in realpath(): " + std::to_string(GetLastError()));
		return path(out.substr(0, length));
#endif
	}

	bool exists() const {
#if defined(_WIN32)
		return GetFileAttributesW(wstr().c_str()) != INVALID_FILE_ATTRIBUTES;
#else
		struct stat sb;
		return stat(str().c_str(), &sb) == 0;
#endif
	}

	size_t file_size() const {
#if defined(_WIN32)
		struct _stati64 sb;
		if (_wstati64(wstr().c_str(), &sb) != 0)
			throw std::runtime_error("path::file_size(): cannot stat file \"" + str() + "\"!");
#else
		struct stat sb;
		if (stat(str().c_str(), &sb) != 0)
			throw std::runtime_error("path::file_size(): cannot stat file \"" + str() + "\"!");
#endif
		return (size_t) sb.st_size;
	}

	bool is_directory() const {
#if defined(_WIN32)
		DWORD result = GetFileAttributesW(wstr().c_str());
		if (result == INVALID_FILE_ATTRIBUTES)
			return false;
		return (result & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
		struct stat sb;
		if (stat(str().c_str(), &sb))
			return false;
		return S_ISDIR(sb.st_mode);
#endif
	}

	bool is_file() const {
#if defined(_WIN32)
		DWORD attr = GetFileAttributesW(wstr().c_str());
		return (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY) == 0);
#else
		struct stat sb;
		if (stat(str().c_str(), &sb))
			return false;
		return S_ISREG(sb.st_mode);
#endif
	}

	std::string filename() const {
		if (empty())
			return "";
		const std::string &last = m_path[m_path.size()-1];
		return last;
	}

	std::string extension() const {
		const std::string &name = filename();
		size_t pos = name.find_last_of(".");
		if (pos == std::string::npos)
			return "";
		return name.substr(pos+1);
	}

	std::string basename() const {
		const std::string &name = filename();
		size_t pos = name.find_last_of(".");
		if (pos == std::string::npos)
			return name;
		return name.substr(0, pos);
	}

	path parent_path() const {
		path result;
		result.m_absolute = m_absolute;
		result.m_smb = m_smb;

		if (m_path.empty()) {
			if (!m_absolute)
				result.m_path.push_back("..");
		} else {
			size_t until = m_path.size() - 1;
			for (size_t i = 0; i < until; ++i)
				result.m_path.push_back(m_path[i]);
		}
		return result;
	}

	path stem() const {
		return parent_path()/basename();
	}

	path with_extension(const std::string& ext) const {
		return parent_path()/(basename()+"."+ext);
	}

	path operator/(const path &other) const {
		if (other.m_absolute)
			throw std::runtime_error("path::operator/(): expected a relative path!");
		if (m_type != other.m_type)
			throw std::runtime_error("path::operator/(): expected a path of the same type!");

		path result(*this);

		for (size_t i=0; i<other.m_path.size(); ++i)
			result.m_path.push_back(other.m_path[i]);

		return result;
	}

	std::string str(path_type type = native_path) const {
		std::ostringstream oss;

		if (m_absolute) {
			if (m_type == posix_path)
				oss << "/";
			else {
				size_t length = 0;
				for (size_t i = 0; i < m_path.size(); ++i)
					// No special case for the last segment to count the NULL character
					length += m_path[i].length() + 1;
				if (m_smb)
					length += 2;

				// Windows requires a \\?\ prefix to handle paths longer than MAX_PATH
				// (including their null character). NOTE: relative paths >MAX_PATH are
				// not supported at all in Windows.
				if (length > MAX_PATH_WINDOWS_LEGACY) {
					if (m_smb)
						oss << "\\\\?\\UNC\\";
					else
						oss << "\\\\?\\";
				} else if (m_smb)
					oss << "\\\\";
			}
		}

		for (size_t i=0; i<m_path.size(); ++i) {
			oss << m_path[i];
			if (i+1 < m_path.size()) {
				if (type == posix_path)
					oss << '/';
				else
					oss << '\\';
			}
		}

		return oss.str();
	}

	void set(const std::string &str, path_type type = native_path) {
		m_type = type;
		if (type == windows_path) {
			std::string tmp = str;

			// Long windows paths (sometimes) begin with the prefix \\?\. It should only
			// be used when the path is >MAX_PATH characters long, so we remove it
			// for convenience and add it back (if necessary) in str()/wstr().
			static const std::string LONG_PATH_PREFIX = "\\\\?\\";
			if (tmp.length() >= LONG_PATH_PREFIX.length()
			 && std::mismatch(std::begin(LONG_PATH_PREFIX), std::end(LONG_PATH_PREFIX), std::begin(tmp)).first == std::end(LONG_PATH_PREFIX)) {
				tmp.erase(0, LONG_PATH_PREFIX.length());
			}

			// Special-case handling of absolute SMB paths, which start with the prefix "\\".
			if (tmp.length() >= 2 && tmp[0] == '\\' && tmp[1] == '\\') {
				m_path = {};
				tmp.erase(0, 2);

				// Interestingly, there is a special-special case where relative paths may be specified as beginning with a "\\"
				// when a non-SMB file with a more-than-260-characters-long absolute _local_ path is double-clicked. This seems to
				// only happen with single-segment relative paths, so we can check for this condition by making sure no further
				// path separators are present.
				if (tmp.find_first_of("/\\") != std::string::npos)
					m_absolute = m_smb = true;
				else
					m_absolute = m_smb = false;

			// Special-case handling of absolute SMB paths, which start with the prefix "UNC\"
			} else if (tmp.length() >= 4 && tmp[0] == 'U' && tmp[1] == 'N' && tmp[2] == 'C' && tmp[3] == '\\') {
				m_path = {};
				tmp.erase(0, 4);
				m_absolute = true;
				m_smb = true;
			// Special-case handling of absolute local paths, which start with the drive letter and a colon "X:\"
			// So that UTF-8 works, do not call std::isalpha if the high bit is set, as that causes an assert on Windows.
			} else if (tmp.length() >= 3 && ((unsigned char)tmp[0] < 0x80) && std::isalpha(tmp[0]) &&
					   tmp[1] == ':' && (tmp[2] == '\\' || tmp[2] == '/')) {
				m_path = {tmp.substr(0, 2)};
				tmp.erase(0, 3);
				m_absolute = true;
				m_smb = false;
			// Relative path
			} else {
				m_path = {};
				m_absolute = false;
				m_smb = false;
			}

			std::vector<std::string> tokenized = tokenize(tmp, "/\\");
			m_path.insert(std::end(m_path), std::begin(tokenized), std::end(tokenized));
		} else {
			m_path = tokenize(str, "/");
			m_absolute = !str.empty() && str[0] == '/';
		}
	}

	path &operator=(const path &path) {
		m_type = path.m_type;
		m_path = path.m_path;
		m_absolute = path.m_absolute;
		m_smb = path.m_smb;
		return *this;
	}

	path &operator=(path &&path) {
		if (this != &path) {
			m_type = path.m_type;
			m_path = std::move(path.m_path);
			m_absolute = path.m_absolute;
			m_smb = path.m_smb;
		}
		return *this;
	}

	friend std::ostream &operator<<(std::ostream &os, const path &path) {
		os << path.str();
		return os;
	}

	bool remove_file() const {
#if !defined(_WIN32)
		return std::remove(str().c_str()) == 0;
#else
		if (is_directory()) {
			return RemoveDirectoryW(wstr().c_str()) != 0;
		} else {
			return DeleteFileW(wstr().c_str()) != 0;
		}
#endif
	}

	bool resize_file(size_t target_length) {
#if !defined(_WIN32)
		return ::truncate(str().c_str(), (off_t) target_length) == 0;
#else
		HANDLE handle = CreateFileW(wstr().c_str(), GENERIC_WRITE, 0, nullptr, 0, FILE_ATTRIBUTE_NORMAL, nullptr);
		if (handle == INVALID_HANDLE_VALUE)
			return false;
		LARGE_INTEGER size;
		size.QuadPart = (LONGLONG) target_length;
		if (SetFilePointerEx(handle, size, NULL, FILE_BEGIN) == 0) {
			CloseHandle(handle);
			return false;
		}
		if (SetEndOfFile(handle) == 0) {
			CloseHandle(handle);
			return false;
		}
		CloseHandle(handle);
		return true;
#endif
	}

	static path getcwd() {
#if !defined(_WIN32)
		char temp[PATH_MAX];
		if (::getcwd(temp, PATH_MAX) == NULL)
			throw std::runtime_error("Internal error in getcwd(): " + std::string(strerror(errno)));
		return path(temp);
#else
		std::wstring temp(MAX_PATH_WINDOWS, '\0');
		if (!_wgetcwd(&temp[0], MAX_PATH_WINDOWS))
			throw std::runtime_error("Internal error in getcwd(): " + std::to_string(GetLastError()));
		return path(temp.c_str());
#endif
	}

#if defined(_WIN32)
	std::wstring wstr(path_type type = native_path) const {
		std::string temp = str(type);
		int size = MultiByteToWideChar(CP_UTF8, 0, &temp[0], (int)temp.size(), NULL, 0);
		std::wstring result(size, 0);
		MultiByteToWideChar(CP_UTF8, 0, &temp[0], (int)temp.size(), &result[0], size);
		return result;
	}


	void set(const std::wstring &wstring, path_type type = native_path) {
		std::string string;
		if (!wstring.empty()) {
			int size = WideCharToMultiByte(CP_UTF8, 0, &wstring[0], (int)wstring.size(),
							NULL, 0, NULL, NULL);
			string.resize(size, 0);
			WideCharToMultiByte(CP_UTF8, 0, &wstring[0], (int)wstring.size(),
								&string[0], size, NULL, NULL);
		}
		set(string, type);
	}

	path &operator=(const std::wstring &str) { set(str); return *this; }
#endif

	bool operator==(const path &p) const { return p.m_path == m_path; }
	bool operator!=(const path &p) const { return p.m_path != m_path; }

protected:
	static std::vector<std::string> tokenize(const std::string &string, const std::string &delim) {
		std::string::size_type lastPos = 0, pos = string.find_first_of(delim, lastPos);
		std::vector<std::string> tokens;

		while (lastPos != std::string::npos) {
			if (pos != lastPos)
				tokens.push_back(string.substr(lastPos, pos - lastPos));
			lastPos = pos;
			if (lastPos == std::string::npos || lastPos + 1 == string.length())
				break;
			pos = string.find_first_of(delim, ++lastPos);
		}

		return tokens;
	}

protected:
#if defined(_WIN32)
	static const size_t MAX_PATH_WINDOWS = 32767;
#endif
	static const size_t MAX_PATH_WINDOWS_LEGACY = 260;
	path_type m_type;
	std::vector<std::string> m_path;
	bool m_absolute;
	bool m_smb; // Unused, except for on Windows
};

inline bool create_directory(const path& p) {
#if defined(_WIN32)
	return CreateDirectoryW(p.wstr().c_str(), NULL) != 0;
#else
	return mkdir(p.str().c_str(), S_IRWXU) == 0;
#endif
}

inline bool create_directories(const path& p) {
#if defined(_WIN32)
	return SHCreateDirectory(nullptr, p.make_absolute().wstr().c_str()) == ERROR_SUCCESS;
#else
	if (create_directory(p.str().c_str()))
		return true;

	if (p.empty())
		return false;

	if (errno == ENOENT) {
		if (create_directory(p.parent_path()))
			return mkdir(p.str().c_str(), S_IRWXU) == 0;
		else
			return false;
	}
	return false;
#endif
}

NAMESPACE_END(filesystem)
