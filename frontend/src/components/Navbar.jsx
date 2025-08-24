import React, { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { 
  TrendingUp, 
  Search, 
  PieChart, 
  Eye, 
  BarChart3, 
  User, 
  LogOut,
  Menu,
  X,
  Brain,
  Newspaper
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import toast from 'react-hot-toast'

const Navbar = () => {
  const { user, isAuthenticated, logout } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  const handleLogout = async () => {
    try {
      await logout()
      toast.success('Logged out successfully')
      navigate('/')
    } catch (error) {
      toast.error('Error logging out')
    }
  }

  const navItems = [
    { to: '/', label: 'Home', icon: TrendingUp },
    { to: '/ai-predictor', label: 'AI Predictor', icon: Brain },
    { to: '/compare', label: 'Compare', icon: BarChart3 },
    { to: '/news', label: 'News', icon: Newspaper },
  ]

  const authenticatedNavItems = [
    { to: '/portfolio', label: 'Portfolio', icon: PieChart },
    { to: '/watchlist', label: 'Watchlist', icon: Eye },
  ]

  const isActivePath = (path) => {
    return location.pathname === path
  }

  const NavLink = ({ to, children, className = '', mobile = false }) => (
    <Link
      to={to}
      className={`
        ${className}
        ${isActivePath(to) 
          ? 'text-primary-600 bg-primary-50' 
          : 'text-gray-600 hover:text-primary-600 hover:bg-gray-50'
        }
        ${mobile 
          ? 'block px-3 py-2 rounded-md text-base font-medium' 
          : 'px-3 py-2 rounded-md text-sm font-medium'
        }
        transition-colors duration-200
      `}
      onClick={() => setIsMobileMenuOpen(false)}
    >
      {children}
    </Link>
  )

  return (
    <nav className="bg-white shadow-lg border-b border-gray-200 sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link 
            to="/" 
            className="flex items-center space-x-2 font-bold text-xl text-primary-600"
          >
            <TrendingUp className="h-8 w-8" />
            <span>Stock Analyzer</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon
              return (
                <NavLink key={item.to} to={item.to} className="flex items-center space-x-1">
                  <Icon className="h-4 w-4" />
                  <span>{item.label}</span>
                </NavLink>
              )
            })}

            {isAuthenticated && authenticatedNavItems.map((item) => {
              const Icon = item.icon
              return (
                <NavLink key={item.to} to={item.to} className="flex items-center space-x-1">
                  <Icon className="h-4 w-4" />
                  <span>{item.label}</span>
                </NavLink>
              )
            })}
          </div>

          {/* Desktop Auth Section */}
          <div className="hidden md:flex items-center space-x-4">
            {isAuthenticated ? (
              <div className="flex items-center space-x-3">
                <span className="text-sm text-gray-700">
                  Welcome, {user?.username}
                </span>
                <button
                  onClick={handleLogout}
                  className="flex items-center space-x-1 px-3 py-2 text-sm font-medium text-gray-600 hover:text-red-600 transition-colors"
                >
                  <LogOut className="h-4 w-4" />
                  <span>Logout</span>
                </button>
              </div>
            ) : (
              <div className="flex items-center space-x-2">
                <Link
                  to="/login"
                  className="px-4 py-2 text-sm font-medium text-primary-600 hover:text-primary-700 transition-colors"
                >
                  Login
                </Link>
                <Link
                  to="/register"
                  className="px-4 py-2 text-sm font-medium bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                >
                  Sign Up
                </Link>
              </div>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-600 hover:text-primary-600 hover:bg-gray-50 transition-colors"
            >
              {isMobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 border-t border-gray-200">
              {navItems.map((item) => {
                const Icon = item.icon
                return (
                  <NavLink key={item.to} to={item.to} mobile className="flex items-center space-x-2">
                    <Icon className="h-5 w-5" />
                    <span>{item.label}</span>
                  </NavLink>
                )
              })}

              {isAuthenticated && authenticatedNavItems.map((item) => {
                const Icon = item.icon
                return (
                  <NavLink key={item.to} to={item.to} mobile className="flex items-center space-x-2">
                    <Icon className="h-5 w-5" />
                    <span>{item.label}</span>
                  </NavLink>
                )
              })}

              <div className="border-t border-gray-200 pt-3 mt-3">
                {isAuthenticated ? (
                  <div className="space-y-2">
                    <div className="px-3 py-2 text-sm text-gray-700">
                      Welcome, {user?.username}
                    </div>
                    <button
                      onClick={handleLogout}
                      className="flex items-center space-x-2 w-full px-3 py-2 text-base font-medium text-red-600 hover:bg-red-50 rounded-md transition-colors"
                    >
                      <LogOut className="h-5 w-5" />
                      <span>Logout</span>
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Link
                      to="/login"
                      className="block px-3 py-2 text-base font-medium text-primary-600 hover:bg-primary-50 rounded-md transition-colors"
                      onClick={() => setIsMobileMenuOpen(false)}
                    >
                      Login
                    </Link>
                    <Link
                      to="/register"
                      className="block px-3 py-2 text-base font-medium bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors"
                      onClick={() => setIsMobileMenuOpen(false)}
                    >
                      Sign Up
                    </Link>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}

export default Navbar
