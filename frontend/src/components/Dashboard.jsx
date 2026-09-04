import React, { useState, useEffect, useRef, useMemo } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  SquarePen,
  MessageSquare,
  LogOut,
  Download,
  ArrowUp,
  User,
  ExternalLink,
  FileText,
  Save,
  X,
  ChevronsRight,
  Undo,
  Redo,
  Bold,
  Italic,
  Underline,
  Strikethrough,
  List,
  ListOrdered,
  Type,
  AlignLeft,
  AlignCenter,
  AlignRight,
  AlignJustify,
  Highlighter,
  Palette,
  Quote,
  Code,
  Link as LinkIcon,
  Image as ImageIcon,
  Menu,
  MoreVertical,
  Copy,
  Sparkles,
  Upload,
  Loader2,
  RefreshCw,
  Trash2,
  Mic,
  Brain,
  AudioLines
} from 'lucide-react';
import { useEditor, EditorContent } from '@tiptap/react';
import { StarterKit } from '@tiptap/starter-kit';
import { Underline as UnderlineExt } from '@tiptap/extension-underline';
import { TextAlign } from '@tiptap/extension-text-align';
import { Color } from '@tiptap/extension-color';
import { TextStyle } from '@tiptap/extension-text-style';
import { Highlight } from '@tiptap/extension-highlight';
import { Link } from '@tiptap/extension-link';
import { Image } from '@tiptap/extension-image';
import { FontFamily } from '@tiptap/extension-font-family';
import showdown from 'showdown';
import TurndownService from 'turndown';

const converter = new showdown.Converter();
const turndownService = new TurndownService();

const COLORS = [
  '#000000', '#434343', '#666666', '#999999', '#b7b7b7', '#cccccc', '#d9d9d9', '#efefef', '#f3f3f3', '#ffffff',
  '#980000', '#ff0000', '#ff9900', '#ffff00', '#00ff00', '#00ffff', '#4a86e8', '#0000ff', '#9900ff', '#ff00ff',
  '#e6b8af', '#f4cccc', '#fce5cd', '#fff2cc', '#d9ead3', '#d0e0e3', '#c9daf8', '#cfe2f3', '#d9d2e9', '#ead1dc',
  '#dd7e6b', '#ea9999', '#f9cb9c', '#ffe599', '#b6d7a8', '#a2c4c9', '#a4c2f4', '#9fc5e8', '#b4a7d6', '#d5a6bd',
];

const FONTS = [
  { label: 'Default', value: 'Inter, sans-serif' },
  { label: 'Arial', value: 'Arial, Helvetica, sans-serif' },
  { label: 'Arial Black', value: '"Arial Black", Gadget, sans-serif' },
  { label: 'Brush Script', value: '"Brush Script MT", cursive' },
  { label: 'Comic Sans', value: '"Comic Sans MS", cursive' },
  { label: 'Courier New', value: '"Courier New", Courier, monospace' },
  { label: 'Georgia', value: 'Georgia, serif' },
  { label: 'Helvetica', value: 'Helvetica, Arial, sans-serif' },
  { label: 'Impact', value: 'Impact, Charcoal, sans-serif' },
  { label: 'Lucida Console', value: '"Lucida Console", Monaco, monospace' },
  { label: 'Lucida Sans', value: '"Lucida Sans Unicode", "Lucida Grande", sans-serif' },
  { label: 'Palatino', value: '"Palatino Linotype", "Book Antiqua", Palatino, serif' },
  { label: 'Tahoma', value: 'Tahoma, Geneva, sans-serif' },
  { label: 'Times New Roman', value: '"Times New Roman", Times, serif' },
  { label: 'Trebuchet MS', value: '"Trebuchet MS", Helvetica, sans-serif' },
  { label: 'Verdana', value: 'Verdana, Geneva, sans-serif' },
  { label: 'Monospace', value: 'monospace' },
  { label: 'Serif', value: 'serif' },
  { label: 'Cursive', value: 'cursive' },
];

const SIZES = ['8px', '9px', '10px', '11px', '12px', '14px', '16px', '18px', '20px', '24px', '26px', '28px', '36px', '48px', '72px'];

const ColorPickerPopover = ({ onSelect, onClose, current }) => {
  return (
    <div className="color-picker-popover">
      <div className="color-grid">
        {COLORS.map(color => (
          <button
            type="button"
            key={color}
            className={`color-cell ${current === color ? 'active' : ''}`}
            style={{ backgroundColor: color }}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => { onSelect(color); onClose(); }}
            title={color}
          />
        ))}
      </div>
      <button type="button" className="clear-color-btn" onMouseDown={(e) => e.preventDefault()} onClick={() => { onSelect(''); onClose(); }}>
        Clear Color
      </button>
    </div>
  );
};

const MenuBar = ({ editor, filename, onSave, onCancel, isSaving }) => {
  const [showTextColor, setShowTextColor] = useState(false);
  const [showHighlight, setShowHighlight] = useState(false);
  const fileInputRef = useRef(null);

  if (!editor) return null;

  const handleImageUpload = async (event) => {
    const file = event.target.files?.[0];
    if (file && filename) {
      const formData = new FormData();
      formData.append('image', file);

      try {
        const token = localStorage.getItem('token');
        const res = await axios.post(`${API_BASE}/blog/${filename}/upload-image`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
            Authorization: `Bearer ${token}`
          }
        });

        if (res.data.url) {
          // res.data.url is like "/images/..."
          const fullUrl = `${API_BASE}${res.data.url}`;
          editor.chain().focus().setImage({ src: fullUrl }).run();
        }
      } catch (err) {
        console.error('Image upload failed', err);
        alert('Failed to upload image. Please try again.');
      }
    }
  };

  return (
    <div className="tiptap-menu-bar">
      <div className="menu-group">
        <select
          onChange={(e) => editor.chain().focus().setFontFamily(e.target.value).run()}
          className="font-select"
          style={{ minWidth: '120px' }}
          value={editor.getAttributes('textStyle').fontFamily || ''}
        >
          {FONTS.map(font => (
            <option key={font.value} value={font.value}>{font.label}</option>
          ))}
        </select>
        <select
          onChange={(e) => editor.chain().focus().setFontSize(e.target.value).run()}
          className="font-select"
          style={{ minWidth: '60px' }}
          value={editor.getAttributes('textStyle').fontSize || '16px'}
        >
          {SIZES.map(size => (
            <option key={size} value={size}>{size.replace('px', '')}</option>
          ))}
        </select>
      </div>

      <div className="menu-divider" />

      <div className="menu-group">
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().undo().run()} disabled={!editor.can().undo()} className="menu-btn" title="Undo"><Undo size={18} /></button>
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().redo().run()} disabled={!editor.can().redo()} className="menu-btn" title="Redo"><Redo size={18} /></button>
      </div>

      <div className="menu-divider" />

      <div className="menu-group">
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().toggleBold().run()} className={`menu-btn ${editor.isActive('bold') ? 'active' : ''}`} title="Bold"><Bold size={18} /></button>
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().toggleItalic().run()} className={`menu-btn ${editor.isActive('italic') ? 'active' : ''}`} title="Italic"><Italic size={18} /></button>
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().toggleUnderline().run()} className={`menu-btn ${editor.isActive('underline') ? 'active' : ''}`} title="Underline"><Underline size={18} /></button>
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().toggleStrike().run()} className={`menu-btn ${editor.isActive('strike') ? 'active' : ''}`} title="Strike"><Strikethrough size={18} /></button>
      </div>

      <div className="menu-divider" />

      <div className="menu-group">
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()} className={`menu-btn ${editor.isActive('heading', { level: 1 }) ? 'active' : ''}`} title="Heading 1 (H1)">H1</button>
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()} className={`menu-btn ${editor.isActive('heading', { level: 2 }) ? 'active' : ''}`} title="Heading 2 (H2)">H2</button>
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().toggleHeading({ level: 3 }).run()} className={`menu-btn ${editor.isActive('heading', { level: 3 }) ? 'active' : ''}`} title="Heading 3 (H3)">H3</button>
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().toggleBulletList().run()} className={`menu-btn ${editor.isActive('bulletList') ? 'active' : ''}`} title="Bullet List"><List size={18} /></button>
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().toggleOrderedList().run()} className={`menu-btn ${editor.isActive('orderedList') ? 'active' : ''}`} title="Ordered List"><ListOrdered size={18} /></button>
      </div>

      <div className="menu-divider" />

      <div className="menu-group">
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().setTextAlign('left').run()} className={`menu-btn ${editor.isActive({ textAlign: 'left' }) ? 'active' : ''}`} title="Align Left"><AlignLeft size={18} /></button>
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().setTextAlign('center').run()} className={`menu-btn ${editor.isActive({ textAlign: 'center' }) ? 'active' : ''}`} title="Align Center"><AlignCenter size={18} /></button>
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().setTextAlign('right').run()} className={`menu-btn ${editor.isActive({ textAlign: 'right' }) ? 'active' : ''}`} title="Align Right"><AlignRight size={18} /></button>
      </div>

      <div className="menu-divider" />

      <div className="menu-group">
        <div style={{ position: 'relative' }}>
          <button
            onClick={() => { setShowHighlight(!showHighlight); setShowTextColor(false); }}
            className={`menu-btn ${editor.isActive('highlight') ? 'active' : ''}`}
            title="Highlight Color"
          >
            <Highlighter size={18} />
          </button>
          {showHighlight && (
            <ColorPickerPopover
              current={editor.getAttributes('highlight').color}
              onSelect={(color) => editor.chain().focus().setHighlight({ color }).run()}
              onClose={() => setShowHighlight(false)}
            />
          )}
        </div>

        <div style={{ position: 'relative' }}>
          <button
            onClick={() => { setShowTextColor(!showTextColor); setShowHighlight(false); }}
            className="menu-btn"
            title="Text Color"
          >
            <Palette size={18} />
          </button>
          {showTextColor && (
            <ColorPickerPopover
              current={editor.getAttributes('textStyle').color}
              onSelect={(color) => editor.chain().focus().setColor(color).run()}
              onClose={() => setShowTextColor(false)}
            />
          )}
        </div>

        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().toggleBlockquote().run()} className={`menu-btn ${editor.isActive('blockquote') ? 'active' : ''}`} title="Quote"><Quote size={18} /></button>
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => editor.chain().focus().toggleCodeBlock().run()} className={`menu-btn ${editor.isActive('codeBlock') ? 'active' : ''}`} title="Code Block"><Code size={18} /></button>
        <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => fileInputRef.current.click()} className="menu-btn" title="Upload Image">
          <ImageIcon size={18} />
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleImageUpload}
            accept="image/*"
            style={{ display: 'none' }}
          />
        </button>
      </div>

      <div className="menu-divider" />

      <div className="menu-group" style={{ marginLeft: 'auto' }}>
        <button
          onClick={onSave}
          className="menu-btn save-btn"
          disabled={isSaving}
          title="Save Changes"
          style={{ color: '#059669', background: '#ecfdf5' }}
        >
          <Save size={18} />
          <span style={{ marginLeft: '4px', fontSize: '12px' }}>{isSaving ? 'Saving...' : 'Save'}</span>
        </button>
        <button
          onClick={onCancel}
          className="menu-btn cancel-btn"
          disabled={isSaving}
          title="Cancel Editing"
          style={{ color: '#dc2626', background: '#fef2f2' }}
        >
          <X size={18} />
          <span style={{ marginLeft: '4px', fontSize: '12px' }}>Cancel</span>
        </button>
      </div>
    </div>
  );
};

const API_BASE = import.meta.env.VITE_API_BASE || (window.location.hostname === 'localhost' ? 'http://localhost:8000' : '');

const CustomTextStyle = TextStyle.extend({
  addAttributes() {
    return {
      ...this.parent?.(),
      fontWeight: {
        default: null,
        parseHTML: element => element.style.fontWeight,
        renderHTML: attributes => {
          if (!attributes.fontWeight) return {};
          return { style: `font-weight: ${attributes.fontWeight}` };
        },
      },
      fontSize: {
        default: null,
        parseHTML: element => element.style.fontSize,
        renderHTML: attributes => {
          if (!attributes.fontSize) return {};
          return { style: `font-size: ${attributes.fontSize}` };
        },
      },
    }
  },
  addCommands() {
    return {
      setFontWeight: fontWeight => ({ chain }) => {
        return chain().setMark('textStyle', { fontWeight }).run();
      },
      setFontSize: fontSize => ({ chain }) => {
        return chain().setMark('textStyle', { fontSize }).run();
      },
    };
  },
})

const BlogImageWrapper = ({ src, alt, filename, onImageUpdated }) => {
  const [isHovered, setIsHovered] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [isRemoving, setIsRemoving] = useState(false);

  // Extract the image filename from the src (e.g., "images/abc123_ai.webp" → "abc123_ai.webp")
  const imageFile = src.replace(/^.*\/images\//, '').replace(/^images\//, '');

  const handleRegenerate = async () => {
    if (!filename || !imageFile) return;
    setIsRegenerating(true);
    try {
      const token = localStorage.getItem('token');
      const res = await axios.post(`${API_BASE}/blog/${filename}/regenerate-image`, {
        image_filename: imageFile
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.data?.updated_content) {
        onImageUpdated(res.data.updated_content);
      }
    } catch (err) {
      console.error('Image regeneration failed', err);
    } finally {
      setIsRegenerating(false);
    }
  };

  const handleRemove = async () => {
    if (!filename || !imageFile) return;
    setIsRemoving(true);
    try {
      const token = localStorage.getItem('token');
      const res = await axios.post(`${API_BASE}/blog/${filename}/remove-image`, {
        image_filename: imageFile
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.data?.updated_content) {
        onImageUpdated(res.data.updated_content);
      }
    } catch (err) {
      console.error('Image removal failed', err);
    } finally {
      setIsRemoving(false);
    }
  };

  return (
    <div
      className="blog-image-wrapper"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <img src={src} alt={alt || 'blog image'} />
      {isHovered && !isRegenerating && !isRemoving && (
        <div className="blog-image-overlay">
          <button className="img-overlay-btn btn-regenerate" onClick={handleRegenerate} title="Generate a new image">
            <RefreshCw size={16} />
            <span>Regenerate</span>
          </button>
          <button className="img-overlay-btn btn-remove" onClick={handleRemove} title="Remove image and go back to placeholder">
            <Trash2 size={16} />
            <span>Remove</span>
          </button>
        </div>
      )}
      {(isRegenerating || isRemoving) && (
        <div className="blog-image-overlay">
          <div className="img-overlay-loading">
            <Loader2 className="animate-spin" size={22} />
            <span>{isRegenerating ? 'Regenerating...' : 'Removing...'}</span>
          </div>
        </div>
      )}
    </div>
  );
};

const ImagePlaceholderCard = ({ placeholderTag, index, prompt, filename, onImageUpdated }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const handleGenerateAI = async () => {
    if (!filename) {
      setError('Please wait until blog draft is saved before generating images.');
      return;
    }
    setIsGenerating(true);
    setError('');
    try {
      const token = localStorage.getItem('token');
      const res = await axios.post(`${API_BASE}/blog/${filename}/generate-image`, {
        placeholder: placeholderTag,
        prompt: prompt
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (res.data && res.data.updated_content) {
        onImageUpdated(res.data.updated_content);
      }
    } catch (err) {
      console.error('AI image generation failed', err);
      setError(err.response?.data?.detail || 'Failed to generate AI image.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleManualUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!filename) {
      setError('Please wait until blog draft is saved before uploading images.');
      return;
    }

    setIsUploading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('placeholder', placeholderTag);

      const token = localStorage.getItem('token');
      const res = await axios.post(`${API_BASE}/blog/${filename}/upload-image`, formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'multipart/form-data'
        }
      });

      if (res.data && res.data.updated_content) {
        onImageUpdated(res.data.updated_content);
      }
    } catch (err) {
      console.error('Manual upload failed', err);
      setError(err.response?.data?.detail || 'Failed to upload image.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="image-placeholder-card">
      <div className="placeholder-header">
        <div className="placeholder-badge">🎨 Image Slot #{index}</div>
        {prompt && <div className="placeholder-prompt">" {prompt} "</div>}
      </div>

      {(isGenerating || isUploading) ? (
        <div className="placeholder-loading">
          <Loader2 className="animate-spin" size={24} />
          <span>{isGenerating ? 'Generating AI Image...' : 'Uploading Image...'}</span>
        </div>
      ) : (
        <div className="placeholder-actions">
          <button
            type="button"
            className="placeholder-btn btn-ai"
            onClick={handleGenerateAI}
            title="Generate Image using AI"
          >
            <Sparkles size={18} />
            <span>Generate by AI</span>
          </button>

          <button
            type="button"
            className="placeholder-btn btn-upload"
            onClick={handleManualUploadClick}
            title="Upload local image file"
          >
            <Upload size={18} />
            <span>Upload Manually</span>
          </button>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/png, image/jpeg, image/webp, image/gif"
            style={{ display: 'none' }}
          />
        </div>
      )}

      {error && <div className="placeholder-error">{error}</div>}
    </div>
  );
};

const Dashboard = () => {

  const [pastBlogs, setPastBlogs] = useState([]);
  const [selectedBlog, setSelectedBlog] = useState(null);
  const [topic, setTopic] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedBlog, setGeneratedBlog] = useState(null);
  const [error, setError] = useState('');
  const [generatingSteps, setGeneratingSteps] = useState([]);
  const [progress, setProgress] = useState(0); // For polling fallback
  const [user, setUser] = useState(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState('');
  const [editedTitle, setEditedTitle] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [showActionMenu, setShowActionMenu] = useState(false);
  const [isLoadingBlog, setIsLoadingBlog] = useState(false);
  const [linkedinStatus, setLinkedinStatus] = useState({ connected: false, name: '' });
  const [isPostingLinkedIn, setIsPostingLinkedIn] = useState(false);
  const [pendingLinkedInPost, setPendingLinkedInPost] = useState(null);
  const [showAccountMenu, setShowAccountMenu] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(window.innerWidth > 768);
  const textareaRef = useRef(null);
  const menuRef = useRef(null);
  const accountMenuRef = useRef(null);

  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);

  const heroHeadings = useMemo(() => [
    "What would you like to blog about?",
    "Write a detailed guide on modern React...",
    "Draft an exciting story about space travel...",
    "Explain quantum computing to a beginner...",
    "Create a comprehensive tutorial on AI...",
    "Summarize the history of the Roman Empire..."
  ], []);
  const [headingIndex, setHeadingIndex] = useState(0);

  useEffect(() => {
    // Only cycle headings if the user hasn't started typing yet
    if (topic.trim()) return;
    const interval = setInterval(() => {
      setHeadingIndex((prev) => (prev + 1) % heroHeadings.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [topic, heroHeadings]);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => setIsListening(true);
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setTopic(prev => (prev ? prev + ' ' + transcript : transcript).trim());
      };
      recognition.onerror = (event) => {
        console.error("Speech recognition error", event.error);
        setIsListening(false);
        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
          alert("Microphone access is blocked! Please click the microphone icon in your browser's address bar and allow access to use voice input.");
        } else if (event.error === 'network') {
          alert("Network error: Your browser's built-in speech service failed to connect. (Note: Open-source Chromium or Brave on Linux often do not support Web Speech API natively. Please try using official Google Chrome.)");
        } else if (event.error !== 'no-speech') {
          alert(`Speech recognition stopped: ${event.error}`);
        }
      };
      recognition.onend = () => setIsListening(false);

      recognitionRef.current = recognition;
    }
  }, []);

  const toggleVoiceInput = () => {
    console.log("toggleVoiceInput called, recognitionRef:", recognitionRef.current, "isListening:", isListening);
    if (!recognitionRef.current) {
      alert("Voice input is not supported in this browser. Try Chrome or Edge.");
      return;
    }
    if (isListening) {
      recognitionRef.current.stop();
    } else {
      try {
        recognitionRef.current.start();
      } catch (err) {
        console.error("Speech start error:", err);
        alert("Could not start voice input: " + err.message);
      }
    }
  };

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'd') {
        e.preventDefault();
        toggleVoiceInput();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isListening]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (accountMenuRef.current && !accountMenuRef.current.contains(event.target)) {
        setShowAccountMenu(false);
      }
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setShowActionMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const editor = useEditor({
    extensions: [
      StarterKit,
      UnderlineExt,
      TextAlign.configure({ types: ['heading', 'paragraph'] }),
      CustomTextStyle,
      Color,
      Highlight.configure({ multicolor: true }),
      Link.configure({ openOnClick: false }),
      Image,
      FontFamily,
    ],
    content: '',
    onUpdate: ({ editor }) => {
      setEditedContent(editor.getHTML());
    },
  });

  useEffect(() => {
    // Handle 401 Unauthorized globally by clearing stale token and redirecting to login
    // NOTE: This must be registered BEFORE the fetch calls so it intercepts them.
    const interceptor = axios.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response && error.response.status === 401) {
          localStorage.removeItem('token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );

    fetchPastBlogs();
    fetchUserProfile();
    fetchLinkedInStatus();

    // Recover from interrupted generation (user refreshed while generating)
    const pending = localStorage.getItem('pendingGeneration');
    if (pending) {
      try {
        const { topic: pendingTopic, startedAt } = JSON.parse(pending);
        const elapsed = Date.now() - startedAt;
        // If it was started less than 15 minutes ago, try to recover
        if (elapsed < 15 * 60 * 1000) {
          console.log(`Reconnecting to active generation for topic: ${pendingTopic}`);
          // Wait a tick for the component to be fully mounted before calling handleGenerate
          setTimeout(() => {
            handleGenerate(null, pendingTopic);
          }, 100);
        } else {
          // Too old, just clear it
          localStorage.removeItem('pendingGeneration');
        }
      } catch (e) {
        localStorage.removeItem('pendingGeneration');
      }
    }

    // Listen for LinkedIn OAuth popup callback
    const handleMessage = (event) => {
      if (event.data?.type === 'LINKEDIN_CONNECTED') {
        fetchLinkedInStatus();
      }
    };
    window.addEventListener('message', handleMessage);
    return () => {
      axios.interceptors.response.eject(interceptor);
      window.removeEventListener('message', handleMessage);
    };
  }, []);

  const fetchUserProfile = async () => {
    const token = localStorage.getItem('token');
    if (!token) return;
    try {
      const res = await axios.get(`${API_BASE}/me`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setUser(res.data);
    } catch (err) {
      console.error('Failed to fetch user profile', err);
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [topic]);

  const loadBlogByFilename = async (filename, blogsList = pastBlogs) => {
    const token = localStorage.getItem('token');
    if (!token || !filename) return;
    setIsLoadingBlog(true);
    try {
      const res = await axios.get(`${API_BASE}/blog/${filename}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const foundBlog = blogsList.find(b => b.filename === filename);
      const title = foundBlog?.title || filename.replace(/_[a-f0-9]+\.md$/, '').replace(/_/g, ' ');
      setSelectedBlog({ filename, title, content: res.data.content });
      setGeneratedBlog(null);
    } catch (err) {
      console.error('Failed to restore active blog', err);
      localStorage.removeItem('activeBlogFilename');
    } finally {
      setIsLoadingBlog(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('activeBlogFilename');
    localStorage.removeItem('pendingGeneration');
    window.location.reload();
  };

  const startNewChat = () => {
    setSelectedBlog(null);
    setGeneratedBlog(null);
    setTopic('');
    setError('');
    setProgress(0);
    setGeneratingSteps([]);
    setIsGenerating(false);
    localStorage.removeItem('activeBlogFilename');
    localStorage.removeItem('pendingGeneration');
    setIsSidebarOpen(false);
  };

  const handleBlogSelect = async (blog) => {
    const token = localStorage.getItem('token');
    setIsLoadingBlog(true);
    try {
      const res = await axios.get(`${API_BASE}/blog/${blog.filename}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setSelectedBlog({ ...blog, content: res.data.content });
      setGeneratedBlog(null);
      localStorage.setItem('activeBlogFilename', blog.filename);
      setIsSidebarOpen(false);
    } catch (err) {
      console.error('Failed to fetch blog', err);
      setError('Failed to fetch blog content.');
    } finally {
      setIsLoadingBlog(false);
    }
  };

  const fetchPastBlogs = async (preventRestore = false) => {
    const token = localStorage.getItem('token');
    if (!token) return;
    try {
      const res = await axios.get(`${API_BASE}/past-blogs`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setPastBlogs(res.data);

      // Auto-restore open blog after page refresh, but only if not prevented
      const activeFilename = localStorage.getItem('activeBlogFilename');
      if (activeFilename && !preventRestore) {
        loadBlogByFilename(activeFilename, res.data);
      }
    } catch (err) {
      console.error('Failed to fetch past blogs', err);
      setError('Failed to fetch past blogs: ' + err.message);
    }
  };

  const handleGenerate = async (e, overrideTopic = null) => {
    if (e) e.preventDefault();
    const activeTopic = overrideTopic || topic;
    if (!activeTopic.trim() || isGenerating) return;

    if (overrideTopic) setTopic(overrideTopic);
    setIsGenerating(true);
    setGeneratedBlog(null);
    setSelectedBlog(null);
    setError('');

    // Persist generation state so we can recover after page refresh
    localStorage.setItem('pendingGeneration', JSON.stringify({ topic: activeTopic, startedAt: Date.now() }));
    setGeneratingSteps([]);

    let reader = null;
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_BASE}/generate-blog`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ topic: activeTopic })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let done = false;
      let finalData = null;
      let buffer = '';

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop(); // Keep the last incomplete part in the buffer
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.substring(6);
              try {
                const data = JSON.parse(dataStr);
                if (data.type === 'progress') {
                  setGeneratingSteps(prev => {
                    if (!prev.includes(data.step)) {
                      return [...prev, data.step];
                    }
                    return prev;
                  });
                } else if (data.type === 'done') {
                  finalData = data.result;
                } else if (data.type === 'error') {
                  throw new Error(data.detail || 'Error during generation');
                }
              } catch (parseErr) {
                console.error('Error parsing SSE data:', parseErr, dataStr);
              }
            }
          }
        }
      }

      if (finalData) {
        setGeneratedBlog(finalData);
        if (finalData.filename) {
          localStorage.setItem('activeBlogFilename', finalData.filename);
        }
        fetchPastBlogs(true);
      }
    } catch (err) {
      setError('Generation failed. Please try again.');
      console.error(err);
    } finally {
      if (reader) {
        reader.cancel().catch(console.error);
      }
      localStorage.removeItem('pendingGeneration');
      setIsGenerating(false);
      setIsEditing(false); // Reset editing mode on new generation
    }
  };

  const handleEdit = () => {
    const rawContent = generatedBlog?.final || selectedBlog?.content || '';
    const currentTitle = generatedBlog?.title || selectedBlog?.title || '';
    const htmlContent = converter.makeHtml(rawContent);
    setEditedContent(htmlContent);
    setEditedTitle(currentTitle);
    if (editor) {
      editor.commands.setContent(htmlContent);
    }
    setIsEditing(true);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
  };

  const handleSave = async () => {
    const filename = generatedBlog?.filename || selectedBlog?.filename;
    if (!filename) return;

    setIsSaving(true);
    try {
      // Convert HTML back to Markdown
      const markdownContent = turndownService.turndown(editedContent);

      const token = localStorage.getItem('token');
      await axios.put(`${API_BASE}/blog/${filename}`,
        { content: markdownContent, title: editedTitle.trim() },
        { headers: { Authorization: `Bearer ${token}` } }
      );

      // Update local state
      if (generatedBlog) {
        setGeneratedBlog({ ...generatedBlog, title: editedTitle.trim(), final: markdownContent });
      }
      if (selectedBlog) {
        setSelectedBlog({ ...selectedBlog, title: editedTitle.trim(), content: markdownContent });
      }

      fetchPastBlogs(true);

      setIsEditing(false);
      setError('');
    } catch (err) {
      console.error('Failed to save blog', err);
      setError('Failed to save changes.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleDownload = async (filename, type) => {
    if (!filename) {
      setError('Filename not available for download.');
      return;
    }
    const url = type === 'md' ? `${API_BASE}/download-md/${filename}` : `${API_BASE}/download-docx/${filename}`;
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(url, {
        responseType: 'blob',
        headers: { Authorization: `Bearer ${token}` }
      });

      // Get filename from Content-Disposition header
      const contentDisposition = response.headers['content-disposition'] || response.headers['Content-Disposition'];
      let downloadFilename = filename;

      if (contentDisposition) {
        // Try to match filename="filename" or filename=filename
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch && filenameMatch[1]) {
          downloadFilename = filenameMatch[1].replace(/['"]/g, '').trim();
        }
      }

      // Final fallback for extension
      if (type === 'docx' && !downloadFilename.toLowerCase().endsWith('.docx')) {
        downloadFilename = downloadFilename.replace(/\.md$/i, '') + '.docx';
      } else if (type === 'md' && !downloadFilename.toLowerCase().endsWith('.md')) {
        downloadFilename += '.md';
      }

      const blob = new Blob([response.data], { type: response.headers['content-type'] });
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = downloadFilename; // This is the important part

      document.body.appendChild(link);
      link.click();

      // Cleanup
      setTimeout(() => {
        document.body.removeChild(link);
        window.URL.revokeObjectURL(downloadUrl);
      }, 100);

      setError('');
    } catch (err) {
      console.error('Download failed', err);
      setError('Download failed. Please check the console for details.');
    }
  };

  const fetchLinkedInStatus = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await axios.get(`${API_BASE}/linkedin/status`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setLinkedinStatus(res.data);
    } catch (err) {
      console.error('Failed to fetch LinkedIn status', err);
    }
  };

  // Auto-resume posting if the user just successfully connected
  useEffect(() => {
    if (linkedinStatus.connected && pendingLinkedInPost) {
      const filename = pendingLinkedInPost;
      setPendingLinkedInPost(null);
      handleLinkedInPost(filename);
    }
  }, [linkedinStatus.connected, pendingLinkedInPost]);

  const connectLinkedIn = () => {
    const token = localStorage.getItem('token');
    const authUrl = `${API_BASE}/linkedin/auth?token=${encodeURIComponent(token)}`;
    window.open(authUrl, 'linkedin-auth', 'width=600,height=700');
    setShowActionMenu(false);
  };

  const disconnectLinkedIn = async () => {
    try {
      const token = localStorage.getItem('token');
      await axios.post(`${API_BASE}/linkedin/disconnect`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setLinkedinStatus({ connected: false, name: '' });
    } catch (err) {
      console.error('Failed to disconnect LinkedIn', err);
    }
    setShowActionMenu(false);
  };

  const handleLinkedInPost = async (filename) => {
    if (!filename) return;

    const envToken = import.meta.env.VITE_LINKEDIN_TOKEN;
    if (!linkedinStatus.connected && !envToken) {
      setPendingLinkedInPost(filename);
      connectLinkedIn();
      return;
    }

    const confirmed = window.confirm(
      "This will:\n\n" +
      "1. Use AI to summarize your blog into a short LinkedIn post\n" +
      "2. Attach the first image from your blog (if any)\n" +
      "3. Post it directly to your LinkedIn feed\n\n" +
      "Do you want to continue?"
    );
    if (!confirmed) return;

    setIsPostingLinkedIn(true);
    try {
      const token = localStorage.getItem('token');
      const body = envToken && !linkedinStatus.connected ? { access_token: envToken } : {};
      const res = await axios.post(`${API_BASE}/blog/${filename}/post-to-linkedin`, body, {
        headers: { Authorization: `Bearer ${token}` }
      });
      alert(res.data.message || "Successfully posted to LinkedIn!");
    } catch (err) {
      console.error('LinkedIn post failed', err);
      alert(err.response?.data?.detail || "Failed to post to LinkedIn.");
    } finally {
      setIsPostingLinkedIn(false);
    }
    setShowActionMenu(false);
  };

  const handleCopyContent = (silent = false) => {
    const content = generatedBlog?.final || selectedBlog?.content || '';
    if (!content) return;

    navigator.clipboard.writeText(content).then(() => {
      if (!silent) alert('Blog content copied to clipboard!');
    }).catch(err => {
      console.error('Failed to copy: ', err);
      if (!silent) alert('Failed to copy content.');
    });
    setShowActionMenu(false);
  };

  const handleImageUpdated = (newContent) => {
    if (generatedBlog) {
      setGeneratedBlog(prev => prev ? ({ ...prev, final: newContent }) : null);
    }
    if (selectedBlog) {
      setSelectedBlog(prev => prev ? ({ ...prev, content: newContent }) : null);
    }
    fetchPastBlogs(true);
  };

  const renderBlog = (content, filename) => {
    if (!content) return null;

    // Clean stray brackets like `[\n![` or `[ ![` left over from old draft placeholders
    const cleanContent = content
      .replace(/\[\s*(\!\[[^\]]*\]\([^\)]+\))/g, '$1')
      .replace(/\[\s*\n+(\!\[)/g, '\n\n$1');

    const regex = /(\[{2}IMAGE_?(?:PLACEHOLDER_)?\d+(?::.*?)?\]{2})/gi;
    const parts = cleanContent.split(regex);


    if (parts.length === 1) {
      return (
        <div className="blog-output">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              img: ({ node, ...props }) => {
                const src = props.src.startsWith('http') ? props.src : (props.src.startsWith('/') ? `${API_BASE}${props.src}` : `${API_BASE}/${props.src}`);
                return <BlogImageWrapper src={src} alt={props.alt} filename={filename} onImageUpdated={handleImageUpdated} />;
              }
            }}
          >
            {content}
          </ReactMarkdown>
        </div>
      );
    }

    return (
      <div className="blog-output">
        {parts.map((part, idx) => {
          if (!part) return null;
          const match = part.match(/^\[{2}IMAGE_?(?:PLACEHOLDER_)?(\d+)(?::\s*(.*?))?\]{2}$/i);
          if (match) {
            const slotNum = match[1] || (idx + 1);
            const promptText = match[2] || '';
            return (
              <ImagePlaceholderCard
                key={`placeholder-${idx}`}
                placeholderTag={part}
                index={slotNum}
                prompt={promptText}
                filename={filename}
                onImageUpdated={handleImageUpdated}
              />
            );
          }
          return (
            <ReactMarkdown
              key={`md-${idx}`}
              remarkPlugins={[remarkGfm]}
              components={{
                img: ({ node, ...props }) => {
                  const src = props.src.startsWith('http') ? props.src : (props.src.startsWith('/') ? `${API_BASE}${props.src}` : `${API_BASE}/${props.src}`);
                  return <BlogImageWrapper src={src} alt={props.alt} filename={filename} onImageUpdated={handleImageUpdated} />;
                }
              }}
            >
              {part}
            </ReactMarkdown>
          );
        })}
      </div>
    );
  };


  return (
    <div className="dashboard-layout">
      {/* Mobile Overlay */}
      {isSidebarOpen && (
        <div className="mobile-overlay" onClick={() => setIsSidebarOpen(false)} />
      )}
      {/* Sidebar */}
      <div className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '2rem' }}>
          <button className="new-chat-btn" style={{ flex: 1, marginBottom: 0 }} onClick={startNewChat}>
            <SquarePen size={20} />
            <span>New Blog</span>
          </button>
          <button
            className="sidebar-close-btn"
            onClick={() => setIsSidebarOpen(false)}
            title="Close sidebar"
          >
            <ChevronsRight size={20} />
          </button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'rgba(255,255,255,0.5)', margin: '1rem 0 0.5rem 0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>History</div>
          {pastBlogs.length > 0 ? (
            pastBlogs.map((blog) => (
              <div
                key={blog.filename}
                className={`past-blog-item ${selectedBlog?.filename === blog.filename ? 'active' : ''}`}
                onClick={() => handleBlogSelect(blog)}
                title={blog.title}
              >
                {blog.title}
              </div>
            ))
          ) : (
            <div style={{ padding: '0.8rem', fontSize: '0.85rem', color: 'rgba(255,255,255,0.6)', fontStyle: 'italic' }}>
              No blogs found
            </div>
          )}
        </div>

        {/* Account Profile Pill Card & Popover Menu */}
        <div ref={accountMenuRef} style={{ position: 'relative', marginTop: 'auto' }}>
          {showAccountMenu && (
            <div className="account-popover-menu">
              <div className="popover-header">ACCOUNT</div>
              <div className="popover-divider" />
              <button
                className="popover-item danger"
                onClick={() => {
                  setShowAccountMenu(false);
                  handleLogout();
                }}
              >
                <LogOut size={16} />
                <span>Logout</span>
              </button>
            </div>
          )}

          <div
            className="account-pill-card"
            onClick={() => setShowAccountMenu(!showAccountMenu)}
            title="Account & Profile Settings"
          >
            <div className="account-avatar">
              {user?.email?.[0]?.toUpperCase() || <User size={16} />}
            </div>
            <div className="account-details">
              <div className="account-name">
                {user?.email ? user.email.split('@')[0] : 'Sirat'} <span className="account-role">[ADMIN]</span>
              </div>
              <div className="account-email">{user?.email || 'sirat@yopmail.com'}</div>
            </div>
            <div className="account-dots">
              <MoreVertical size={16} />
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content" style={{ position: 'relative' }}>
        {!isSidebarOpen && !selectedBlog && !generatedBlog && (
          <button
            onClick={() => setIsSidebarOpen(true)}
            title="Open sidebar"
            style={{
              position: 'fixed',
              top: '1rem',
              left: '1rem',
              zIndex: 90,
              background: '#ffffff',
              border: '1px solid #e2e8f0',
              boxShadow: '0 2px 5px rgba(0,0,0,0.08)',
              cursor: 'pointer',
              color: '#334155',
              padding: '0.5rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: '8px'
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = '#f8fafc'}
            onMouseLeave={(e) => e.currentTarget.style.background = '#ffffff'}
          >
            <Menu size={22} />
          </button>
        )}
        {error && (
          <div style={{
            background: '#fef2f2',
            color: '#991b1b',
            border: '1px solid #fecaca',
            borderRadius: '8px',
            padding: '12px 16px',
            margin: '1rem 2rem 0 2rem',
            fontSize: '0.9rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
          }}>
            <span> {error}</span>
            <button onClick={() => setError('')} style={{ background: 'none', border: 'none', color: '#991b1b', cursor: 'pointer', fontWeight: 'bold', fontSize: '1.1rem' }}>✕</button>
          </div>
        )}
        {!selectedBlog && !generatedBlog ? (
          <div className="welcome-section" style={isGenerating ? { justifyContent: 'flex-start', paddingBottom: 0 } : {}}>
            {isGenerating ? (
              <>
                {/* Top area: user prompt bubble + loading steps */}
                <div style={{ width: '100%', maxWidth: '800px', flex: 1, display: 'flex', flexDirection: 'column', paddingTop: '2rem' }}>
                  <div style={{ 
                    alignSelf: 'flex-end', 
                    background: '#f3f4f6', 
                    padding: '1rem 1.2rem', 
                    borderRadius: '16px', 
                    fontSize: '1.05rem', 
                    color: '#1a1a1a', 
                    marginBottom: '2.5rem',
                    maxWidth: '80%',
                    lineHeight: '1.5',
                    border: '1px solid #e5e7eb'
                  }}>
                    {topic}
                  </div>
                  <div style={{ 
                    alignSelf: 'flex-start', 
                    display: 'flex', 
                    alignItems: 'center', 
                    color: '#1a1a1a', 
                    fontSize: '0.95rem', 
                    fontWeight: 500,
                    padding: '0.6rem 1rem',
                    borderRadius: '8px',
                    background: '#f0fdf4',
                    border: '1px solid #d1fae5',
                    marginBottom: '1rem'
                  }}>
                    <Loader2 size={18} style={{ marginRight: '10px', animation: 'spin 2s linear infinite', color: '#059669' }} />
                    {(() => {
                      const ALL_STEPS = ["Analyzing topic", "Researching the web", "Planning blog structure", "Writing sections", "Merging content", "Deciding image placeholders"];
                      const matchStep = (stepLabel) => generatingSteps.some(s => s.startsWith(stepLabel));
                      const latestCompletedIndex = ALL_STEPS.reduce((maxIdx, step, i) => matchStep(step) ? i : maxIdx, -1);
                      const currentStepText = ALL_STEPS[latestCompletedIndex + 1] || "Finalizing";
                      const displayText = currentStepText === "Analyzing topic" ? `Analyzing topic: ${topic}` : currentStepText;
                      return <span>{displayText} · {generatingSteps.length} completed</span>;
                    })()}
                    <span style={{ marginLeft: '10px', fontSize: '1.3rem', lineHeight: 1 }}>›</span>
                  </div>

                  {/* Detailed step list */}
                  <div style={{ marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.6rem', paddingLeft: '0.5rem' }}>
                    {(() => {
                      const ALL_STEPS = ["Analyzing topic", "Researching the web", "Planning blog structure", "Writing sections", "Merging content", "Deciding image placeholders"];
                      const matchStep = (stepLabel) => generatingSteps.some(s => s.startsWith(stepLabel));
                      const latestCompletedIndex = ALL_STEPS.reduce((maxIdx, step, i) => matchStep(step) ? i : maxIdx, -1);
                      
                      return ALL_STEPS.map((step, i) => {
                        const isDone = matchStep(step);
                        const isSkipped = !isDone && i < latestCompletedIndex;
                        const isCompleted = isDone || isSkipped;
                        const isActive = i === latestCompletedIndex + 1;
                        
                        // For the analyzing topic step, show the actual topic name
                        const displayLabel = step === "Analyzing topic" ? `Analyzing topic: ${topic}` : step;
                        
                        return (
                          <div key={step} style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.6rem',
                            fontSize: '0.85rem',
                            color: isCompleted ? '#374151' : isActive ? '#111827' : '#6b7280',
                            transition: 'color 0.3s ease'
                          }}>
                            {isCompleted ? (
                              <span style={{ color: '#059669', fontSize: '1rem' }}>✓</span>
                            ) : isActive ? (
                              <Loader2 size={14} style={{ animation: 'spin 2s linear infinite', color: '#059669' }} />
                            ) : (
                              <span style={{ width: '14px', height: '14px', borderRadius: '50%', border: '1.5px solid #9ca3af', display: 'inline-block' }} />
                            )}
                            <span>{displayLabel}</span>
                          </div>
                        );
                      });
                    })()}
                  </div>
                </div>



                <style>{`
                  @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                  }
                `}</style>
              </>
            ) : (
              <h1 className="welcome-title">
                <span key={headingIndex} className="sliding-text">
                  {heroHeadings[headingIndex]}
                </span>
              </h1>
            )}

            {!isGenerating && (
            <div className="hero-input-container">
              <div className="hero-prompt-wrapper">
                <div className="hero-prompt-icon">
                  <Brain size={22} color="#9ca3af" />
                </div>
                <textarea
                  ref={textareaRef}
                  className="hero-prompt-input with-icon"
                  rows="3"
                  placeholder="Ask anything"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleGenerate();
                    }
                  }}
                  disabled={isGenerating}
                />
                <div className="hero-input-actions">
                  <div className="voice-btn-container">
                    <button
                      className={`voice-btn-circle ${isListening ? 'active' : ''}`}
                      onClick={toggleVoiceInput}
                      type="button"
                    >
                      {isListening ? (
                        <div className="animated-wave">
                          <span className="bar"></span>
                          <span className="bar"></span>
                          <span className="bar"></span>
                          <span className="bar"></span>
                          <span className="bar"></span>
                        </div>
                      ) : (
                        <Mic size={20} color="#666" />
                      )}
                    </button>
                    <div className="voice-tooltip">
                      Dictate <span className="shortcut-keys">Ctrl + Shift + D</span>
                    </div>
                  </div>
                  <button
                    className="gen-btn-circle"
                    disabled={isGenerating || !topic.trim()}
                    onClick={handleGenerate}
                    type="button"
                  >
                    <ArrowUp size={20} color="white" />
                  </button>
                </div>
              </div>
            </div>
            )}

            {!isGenerating && (
            <div style={{ fontSize: '0.75rem', color: '#999', textAlign: 'center', marginTop: '1.5rem' }}>
              Blog Writing Agent can make mistakes. Check important info.
            </div>
            )}
          </div>
        ) : (
          <div className="chat-container">
              <>
                <div className="chat-header">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                    {!isSidebarOpen && (
                      <button
                        onClick={() => setIsSidebarOpen(true)}
                        title="Open sidebar"
                        style={{
                          background: 'transparent',
                          border: 'none',
                          cursor: 'pointer',
                          color: '#1a1a1a',
                          padding: '4px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          borderRadius: '6px'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(0,0,0,0.05)'}
                        onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                      >
                        <Menu size={20} />
                      </button>
                    )}
                    <div className="chat-title">
                      {isEditing ? (
                        <input
                          type="text"
                          value={editedTitle}
                          onChange={(e) => setEditedTitle(e.target.value)}
                          className="edit-title-header-input"
                          placeholder="Blog Title..."
                          style={{
                            fontSize: '1rem',
                            fontWeight: 700,
                            padding: '4px 8px',
                            borderRadius: '6px',
                            border: '1px solid #3b82f6',
                            outline: 'none',
                            width: '100%',
                            maxWidth: '300px',
                            color: '#0f172a',
                            background: '#ffffff'
                          }}
                        />
                      ) : (
                        generatedBlog?.title || selectedBlog?.title || "New Blog"
                      )}
                    </div>
                  </div>
                  <div className="chat-actions" ref={menuRef}>

                    <div style={{ position: 'relative' }}>
                      <button
                        className={`action-icon-btn ${showActionMenu ? 'active' : ''}`}
                        onClick={() => setShowActionMenu(!showActionMenu)}
                        title="Actions"
                      >
                        <Menu size={20} />
                      </button>

                      {showActionMenu && (
                        <div className="action-dropdown" style={{ minWidth: '220px' }}>
                          {!isEditing && (
                            <>
                              <button
                                onClick={() => { handleEdit(); setShowActionMenu(false); }}
                                className="dropdown-item"
                              >
                                <SquarePen size={16} />
                                Edit Blog
                              </button>

                              <button
                                onClick={() => handleCopyContent()}
                                className="dropdown-item"
                              >
                                <Copy size={16} />
                                Copy Content
                              </button>

                              <div className="dropdown-divider" style={{ height: '1px', background: '#f1f5f9', margin: '4px 0' }} />
                            </>
                          )}

                          <button
                            onClick={() => { handleDownload(generatedBlog?.filename || selectedBlog?.filename, 'md'); setShowActionMenu(false); }}
                            className="dropdown-item"
                            disabled={isEditing}
                            title={isEditing ? "Save or cancel editing to download" : "Download Markdown"}
                          >
                            <FileText size={16} />
                            Download MD
                          </button>
                          <button
                            onClick={() => { handleDownload(generatedBlog?.filename || selectedBlog?.filename, 'docx'); setShowActionMenu(false); }}
                            className="dropdown-item"
                            disabled={isEditing}
                            title={isEditing ? "Save or cancel editing to download" : "Download DOCX"}
                          >
                            <Download size={16} />
                            Download DOCX
                          </button>
                          <div className="menu-divider" style={{ margin: '4px 0', borderTop: '1px solid #eee' }} />
                          {linkedinStatus.connected ? (
                            <>
                              <div style={{ padding: '6px 12px', fontSize: '0.75rem', color: '#16a34a', fontWeight: 600 }}>
                                LinkedIn: {linkedinStatus.name}
                              </div>
                              <button
                                onClick={() => handleLinkedInPost(generatedBlog?.filename || selectedBlog?.filename)}
                                className="dropdown-item"
                                disabled={isEditing || isPostingLinkedIn}
                                title="Post a summary to LinkedIn"
                              >
                                <ExternalLink size={16} />
                                {isPostingLinkedIn ? 'Posting...' : 'Post to LinkedIn'}
                              </button>
                              <button
                                onClick={disconnectLinkedIn}
                                className="dropdown-item"
                                style={{ color: '#ef4444' }}
                              >
                                <X size={16} />
                                Disconnect LinkedIn
                              </button>
                            </>
                          ) : (
                            <button
                              onClick={connectLinkedIn}
                              className="dropdown-item"
                              style={{ color: '#0077b5' }}
                            >
                              <ExternalLink size={16} />
                              Connect LinkedIn
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {isLoadingBlog ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: '4rem' }}>
                    <div className="status-indicator" style={{ position: 'static', marginBottom: '1rem' }}>
                      <div className="progress-dot active" style={{ animation: 'pulse 1.5s infinite' }} />
                    </div>
                    <div style={{ color: '#666', fontSize: '0.9rem' }}>Loading blog content...</div>
                  </div>
                ) : isEditing ? (
                  <div
                    className="editor-container tiptap-container"
                    style={{
                      position: 'relative',
                      zIndex: 10,
                      opacity: isSaving ? 0.6 : 1,
                      pointerEvents: isSaving ? 'none' : 'auto',
                      transition: 'opacity 0.2s ease'
                    }}
                  >
                    <div style={{ padding: '12px 16px', borderBottom: '1px solid #e2e8f0', background: '#f8fafc', borderRadius: '12px 12px 0 0', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      <label style={{ fontSize: '0.7rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Blog Main Title / Heading
                      </label>
                      <input
                        type="text"
                        value={editedTitle}
                        onChange={(e) => setEditedTitle(e.target.value)}
                        placeholder="Enter blog heading..."
                        style={{
                          width: '100%',
                          fontSize: '1.2rem',
                          fontWeight: 700,
                          color: '#0f172a',
                          border: '1px solid #cbd5e1',
                          borderRadius: '6px',
                          padding: '6px 10px',
                          outline: 'none',
                          background: '#ffffff'
                        }}
                      />
                    </div>
                    <MenuBar
                      editor={editor}
                      filename={generatedBlog?.filename || selectedBlog?.filename}
                      onSave={handleSave}
                      onCancel={handleCancelEdit}
                      isSaving={isSaving}
                    />
                    <EditorContent editor={editor} className="tiptap-editor" />
                  </div>
                ) : (
                  renderBlog(generatedBlog?.final || selectedBlog?.content, generatedBlog?.filename || selectedBlog?.filename)
                )}
              </>
          </div>
        )}

        {/* Input bar moved to welcome screen */}
      </div>
    </div>
  );
};


export default Dashboard;
