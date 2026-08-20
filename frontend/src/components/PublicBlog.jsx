import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './PublicBlog.css';

const API_BASE = import.meta.env.VITE_API_BASE || (window.location.hostname === 'localhost' ? 'http://localhost:8000' : '');

function PublicBlog() {
  const { filename } = useParams();
  const [blog, setBlog] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchBlog = async () => {
      try {
        const res = await axios.get(`${API_BASE}/public/blog/${filename}`);
        setBlog(res.data);
      } catch (err) {
        console.error('Failed to fetch public blog', err);
        setError('Blog not found or could not be loaded.');
      } finally {
        setLoading(false);
      }
    };
    fetchBlog();
  }, [filename]);

  if (loading) {
    return (
      <div className="public-blog-container loading">
        <div className="spinner"></div>
        <p>Loading article...</p>
      </div>
    );
  }

  if (error || !blog) {
    return (
      <div className="public-blog-container error">
        <h2>{error}</h2>
      </div>
    );
  }

  // Pre-process markdown to properly handle image tags with extra newlines if needed
  let displayContent = blog.content;

  return (
    <div className="public-blog-container">
      <article className="public-blog-content">
        <ReactMarkdown>{displayContent}</ReactMarkdown>
      </article>
      <footer className="public-blog-footer">
        <p>Generated with <a href="/">Agentic AI Blog Creator</a></p>
      </footer>
    </div>
  );
}

export default PublicBlog;
