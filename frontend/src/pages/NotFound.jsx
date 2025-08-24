import React from 'react'
import { Link } from 'react-router-dom'

const NotFound = () => {
  return (
    <div className="text-center py-12">
      <div className="max-w-md mx-auto">
        <div className="text-6xl font-bold text-gray-300 mb-4">404</div>
        <h1 className="text-2xl font-bold text-gray-900 mb-4">Page Not Found</h1>
        <p className="text-gray-600 mb-8">
          The page you're looking for doesn't exist.
        </p>
        <Link
          to="/"
          className="btn-primary"
        >
          Go Home
        </Link>
      </div>
    </div>
  )
}

export default NotFound
