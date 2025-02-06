
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";

const PostDetail = () => {
  const { slug } = useParams(); // ✅ Get the slug from URL
  const [post, setPost] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`http://127.0.0.1:8000/posts/${slug}/`) // ✅ Fetch post by slug
      .then(response => {
        setPost(response.data);
        setLoading(false);
      })
      .catch(error => {
        console.error("Error fetching post:", error);
        setLoading(false);
      });
  }, [slug]);

  if (loading) {
    return <p className="text-center mt-8">Loading post details...</p>;
  }

  if (!post) {
    return <p className="text-center mt-8 text-red-500">Post not found</p>;
  }

  return (
    <div className="max-w-2xl mx-auto bg-white shadow-lg p-6 rounded-lg mt-8">
      {post.featured_image && (
        <img src={post.featured_image} alt={post.title} className="w-full h-64 object-cover rounded-md" />
      )}
      <h2 className="text-3xl font-bold text-gray-800 mt-4">{post.title}</h2>
      <p className="text-gray-600 mt-2"><strong>Author:</strong> {post.author}</p>
      <p className="text-gray-500 text-sm mt-2"><strong>Posted on:</strong> {new Date(post.created_at).toLocaleString()}</p>
      <div className="mt-4 text-gray-700 border-t pt-4">{post.content}</div>
    </div>
  );
};

export default PostDetail;
