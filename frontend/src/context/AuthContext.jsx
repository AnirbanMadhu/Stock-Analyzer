import React, { createContext, useContext, useState, useEffect } from 'react'
import { authApi } from '../services/api'

const AuthContext = createContext()

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  // Check if user is authenticated on app start
  useEffect(() => {
    checkAuth()
  }, [])

  const checkAuth = async () => {
    try {
      const response = await authApi.checkAuth()
      if (response.data.authenticated) {
        setUser(response.data.user)
      }
    } catch (error) {
      console.log('Not authenticated')
    } finally {
      setLoading(false)
    }
  }

  const login = async (credentials) => {
    try {
      const response = await authApi.login(credentials)
      setUser(response.data.user)
      return response.data
    } catch (error) {
      throw error
    }
  }

  const register = async (userData) => {
    try {
      const response = await authApi.register(userData)
      setUser(response.data.user)
      return response.data
    } catch (error) {
      throw error
    }
  }

  const logout = async () => {
    try {
      await authApi.logout()
    } catch (error) {
      console.log('Logout error:', error)
    } finally {
      setUser(null)
    }
  }

  const value = {
    user,
    loading,
    login,
    register,
    logout,
    checkAuth,
    isAuthenticated: !!user
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}
